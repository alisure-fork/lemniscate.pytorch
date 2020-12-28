import os
import math
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from alisuretool.Tools import Tools
from cifar_10_tool import HCBasicBlock
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from cifar_10_tool import FeatureName, Normalize


class HCResNet(nn.Module):

    def __init__(self, block, num_blocks, low_dim=512, linear_bias=True, input_size=32, conv1_stride=1, is_vis=False):
        super(HCResNet, self).__init__()
        self.in_planes = 64
        self.is_vis = is_vis
        self.input_size = input_size

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=conv1_stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2 if self.input_size > 32 else 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear_1024 = nn.Linear(512 * block.expansion, low_dim, bias=linear_bias)
        self.l2norm = Normalize(2)
        pass

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        convB1 = F.relu(self.bn1(self.conv1(x)))
        convB2 = self.layer1(convB1)
        convB3 = self.layer2(convB2)
        convB4 = self.layer3(convB3)
        convB5 = self.layer4(convB4)
        avgPool = F.adaptive_avg_pool2d(convB5, (1, 1))
        avgPool = avgPool.view(avgPool.size(0), -1)
        out_l2norm0 = self.l2norm(avgPool)

        out_logits = self.linear_1024(avgPool)
        out_l2norm = self.l2norm(out_logits)

        feature_dict = {}
        if self.is_vis:
            feature_dict[FeatureName.x] = x
            # feature_dict[FeatureName.ConvB1] = convB1
            # feature_dict[FeatureName.ConvB2] = convB2
            feature_dict[FeatureName.ConvB3] = convB3
            feature_dict[FeatureName.ConvB4] = convB4
            feature_dict[FeatureName.ConvB5] = convB5
            feature_dict[FeatureName.AvgPool] = avgPool
            pass
        feature_dict[FeatureName.Logits0] = avgPool
        feature_dict[FeatureName.L2norm0] = out_l2norm0
        feature_dict[FeatureName.Logits1] = out_logits
        feature_dict[FeatureName.L2norm1] = out_l2norm

        return feature_dict

    pass


class STL10Instance(datasets.STL10):

    def __getitem__(self, index):
        if hasattr(self, "data") and hasattr(self, "labels"):
            if self.labels is not None:
                img, target = self.data[index], int(self.labels[index])
            else:
                img, target = self.data[index], None
        else:
            if self.train:
                img, target = self.train_data[index], self.train_labels[index]
            else:
                img, target = self.test_data[index], self.test_labels[index]
            pass

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        img_transform = img
        if self.transform is not None:
            img_transform = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        img = transforms.Compose([transforms.ToTensor()])(img)

        return img, img_transform, target, index

    @staticmethod
    def data(data_root, batch_size=128, input_size=32):
        Tools.print('==> Preparing data..')

        transform_test = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        class_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        train_set = STL10Instance(root=data_root, split="unlabeled", download=True, transform=transform_test)
        test_train_set = STL10Instance(root=data_root, split="train", download=True, transform=transform_test)
        test_test_set = STL10Instance(root=data_root, split="test", download=True, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2)
        test_train_loader = torch.utils.data.DataLoader(test_train_set, batch_size, shuffle=False, num_workers=2)
        test_test_loader = torch.utils.data.DataLoader(test_test_set, batch_size, shuffle=False, num_workers=2)
        return train_set, train_loader, test_train_set, test_train_loader, test_test_set, test_test_loader, class_name

    pass


class HCRunner(object):

    def __init__(self, low_dim=512, batch_size=128, input_size=32, conv1_stride=1,
                 linear_bias=True, data_root='./data'):
        self.data_root = data_root
        self.batch_size = batch_size

        self.low_dim = low_dim
        self.low_dim_list = [512, self.low_dim]
        self.input_size = input_size
        self.conv1_stride = conv1_stride

        self.linear_bias = linear_bias

        (self.train_set, self.train_loader, self.test_train_set, self.test_train_loader,
         self.test_test_set, self.test_test_loader, self.class_name) = STL10Instance.data(
            self.data_root, batch_size=self.batch_size, input_size=self.input_size)

        self.train_num = self.train_set.__len__()

        self.net = HCResNet(HCBasicBlock, [2, 2, 2, 2], self.low_dim, linear_bias=linear_bias,
                            input_size=self.input_size, conv1_stride=self.conv1_stride).cuda()
        pass

    def load_checkpoint(self, checkpoint_path):
        Tools.print('==> Pre train from checkpoint {} ..'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        checkpoint_value = {key.replace("module.", ""): checkpoint["net"][key]
                            for key in checkpoint["net"].keys()}

        self.net.load_state_dict(checkpoint_value, strict=True)
        Tools.print("{} {}".format(checkpoint['acc'], checkpoint['epoch']))
        pass

    def vis(self, vis_dir, split="unlabeled"):
        self.net.eval()
        loader = self.train_loader
        loader = self.test_train_loader if split == "train" else loader
        loader = self.test_test_loader if split == "test" else loader

        for batch_idx, (images, inputs, labels, indexes) in tqdm(enumerate(loader)):
            inputs, indexes = inputs.cuda(), indexes.cuda()
            feature_dict = self.net(inputs)
            # ic_out_logits = feature_dict[FeatureName.Logits1]
            ic_out_logits = feature_dict[FeatureName.L2norm1]

            image_data = np.asarray(images.permute(0, 2, 3, 1) * 255, np.uint8)
            cluster_id = np.asarray(torch.argmax(ic_out_logits, -1).cpu())
            for i in range(len(indexes)):
                result_path = Tools.new_dir(os.path.join(vis_dir, split, str(cluster_id[i])))
                score = 100 + int(100 * ic_out_logits[i][cluster_id[i]])  # labels[i]
                Image.fromarray(image_data[i]).save(os.path.join(result_path, "{}_{}.png".format(score, indexes[i])))
                pass
            pass
        pass

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    _input_size = 96
    _conv1_stride = 1  # 1
    _low_dim = 128
    _batch_size = 32
    _linear_bias = False
    _checkpoint_path = "./checkpoint/stl10/stl_10_class_128_1level_1600_no_32_1_l1_sum_0_1_96_1/ckpt.t7"
    _vis_dir = "/mnt/4T/ALISURE/Unsupervised/vis/stl10_3"

    Tools.print()
    runner = HCRunner(low_dim=_low_dim, linear_bias=_linear_bias,  input_size=_input_size,
                      conv1_stride=_conv1_stride, batch_size=_batch_size)
    runner.load_checkpoint(_checkpoint_path)
    Tools.print()
    runner.vis(_vis_dir, split="unlabeled")
    runner.vis(_vis_dir, split="train")
    runner.vis(_vis_dir, split="test")
    pass
