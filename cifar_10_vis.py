import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from alisuretool.Tools import Tools
import torchvision.datasets as data_set
import torchvision.transforms as transforms
from cifar_10_3level_no_memory_l2_sum import HCBasicBlock


class CIFAR10Instance(data_set.CIFAR10):

    def __getitem__(self, index):
        if hasattr(self, "data") and hasattr(self, "targets"):
            img, target = self.data[index], self.targets[index]
        else:
            if self.train:
                img, target = self.train_data[index], self.train_labels[index]
            else:
                img, target = self.test_data[index], self.test_labels[index]
            pass

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    @staticmethod
    def data(data_root, is_shuffle=True, batch_size=32):
        Tools.print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = CIFAR10Instance(root=data_root, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=is_shuffle, num_workers=2)
        test_set = CIFAR10Instance(root=data_root, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=is_shuffle, num_workers=2)

        class_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return train_set, train_loader, test_set, test_loader, class_name

    pass


class Classifier(nn.Module):

    def __init__(self, low_dim, linear_bias=True):
        super(Classifier, self).__init__()
        self.dhc = HCResNet(HCBasicBlock, [2, 2, 2, 2], *low_dim, linear_bias=linear_bias).cuda()
        pass

    def forward(self, inputs):
        out = self.dhc(inputs)
        return out[-1]

    pass


class ClassierRunner(object):

    def __init__(self, net, checkpoint_path="./classier.t7", data_root='./data'):
        self.checkpoint_path = checkpoint_path
        self.data_root = data_root
        self.best_acc = 0

        self.train_set, self.train_loader, self.test_set, self.test_loader, _ = CIFAR10Instance.data(
            self.data_root, is_shuffle=True)
        self.train_num = self.train_set.__len__()

        self.net = net.cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self._load_model(self.net)
        pass

    def _load_model(self, net):
        Tools.print('==> Resuming from checkpoint {} ..'.format(self.checkpoint_path))
        checkpoint = torch.load(self.checkpoint_path)

        checkpoint_value = {key.replace("module.", "module.dhc."): checkpoint["net"][key]
                            for key in checkpoint["net"].keys()}
        net.load_state_dict(checkpoint_value)
        best_acc = checkpoint['acc']
        best_epoch = checkpoint['epoch']
        Tools.print("{} {}".format(best_acc, best_epoch))
        pass

    def vis(self, is_test=True, result_file=""):
        self.net.eval()
        loader = self.test_loader if is_test else self.train_loader

        feature = {}
        label = []
        for batch_idx, (inputs, targets, _) in enumerate(loader):

            if batch_idx % 10 == 0:
                Tools.print("{} {}/{}".format(is_test, batch_idx, len(loader)))

            if batch_idx < 500:
                inputs = inputs.cuda()
                outputs = self.net(inputs)
                for key in outputs:
                    if key not in feature:
                        feature[key] = []

                    out_key = outputs[key].cpu().detach().numpy()
                    feature[key].extend(out_key)
                    pass
                label.extend(np.asarray(targets))
                pass
            pass

        Tools.write_to_pkl(result_file, {"feature": feature, "label": label})
        Tools.print("write to {}".format(result_file))
        pass

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    _low_dim = [1024, 512, 256, 128, 64]
    _checkpoint_path = "11_class_1024_5level_512_256_128_64_no_1600_32_1_l1_sum_0_54321"
    _checkpoint_path_classier = "./checkpoint/{}/ckpt.t7".format(_checkpoint_path)
    from cifar_10_5level_no_memory_l2_sum import HCResNet

    Tools.print()
    Tools.print("classier={}".format(_checkpoint_path_classier))

    _net = Classifier(low_dim=_low_dim, linear_bias=False)
    runner = ClassierRunner(net=_net, checkpoint_path=_checkpoint_path_classier)

    Tools.print()
    runner.vis(is_test=True, result_file="./checkpoint/{}/feature_test.pkl".format(_checkpoint_path))
    runner.vis(is_test=False, result_file="./checkpoint/{}/feature_train.pkl".format(_checkpoint_path))
    pass
