import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from alisuretool.Tools import Tools
from cifar_10_tool import HCBasicBlock, ProduceClass, FeatureName, Normalize, CIFAR10Instance


class HCResNet(nn.Module):

    def __init__(self, block, num_blocks, low_dim=512, low_dim2=128,
                 low_dim3=10, low_dim4=10, low_dim5=10, linear_bias=True, is_vis=False):
        super(HCResNet, self).__init__()
        self.in_planes = 64
        self.is_vis = is_vis

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear_1024 = nn.Linear(512 * block.expansion, low_dim, bias=linear_bias)
        self.linear_512 = nn.Linear(low_dim, low_dim2, bias=linear_bias)
        self.linear_256 = nn.Linear(low_dim2, low_dim3, bias=linear_bias)
        self.linear_128 = nn.Linear(low_dim3, low_dim4, bias=linear_bias)
        self.linear_64 = nn.Linear(low_dim4, low_dim5, bias=linear_bias)
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
        avgPool = F.avg_pool2d(convB5, 4)
        avgPool = avgPool.view(avgPool.size(0), -1)
        out_l2norm0 = self.l2norm(avgPool)

        out_logits = self.linear_1024(avgPool)
        out_l2norm = self.l2norm(out_logits)

        out_logits2 = self.linear_512(out_logits)
        out_l2norm2 = self.l2norm(out_logits2)

        out_logits3 = self.linear_256(out_logits2)
        out_l2norm3 = self.l2norm(out_logits3)

        out_logits4 = self.linear_128(out_logits3)
        out_l2norm4 = self.l2norm(out_logits4)

        out_logits5 = self.linear_64(out_logits4)
        out_l2norm5 = self.l2norm(out_logits5)

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
        feature_dict[FeatureName.Logits2] = out_logits2
        feature_dict[FeatureName.L2norm2] = out_l2norm2
        feature_dict[FeatureName.Logits3] = out_logits3
        feature_dict[FeatureName.L2norm3] = out_l2norm3
        feature_dict[FeatureName.Logits4] = out_logits4
        feature_dict[FeatureName.L2norm4] = out_l2norm4
        feature_dict[FeatureName.Logits5] = out_logits5
        feature_dict[FeatureName.L2norm5] = out_l2norm5

        return feature_dict

    pass


class HCRunner(object):

    def __init__(self, low_dim=512, low_dim2=128, low_dim3=10, low_dim4=10, low_dim5=10,
                 ratio1=3, ratio2=2, ratio3=1, ratio4=1, ratio5=1, batch_size=128,
                 is_loss_sum=False, is_adjust_lambda=False, l1_lambda=0.1, learning_rate=0.03,
                 linear_bias=True, has_l1=False, max_epoch=1000, t_epoch=300, first_epoch=200,
                 resume=False, checkpoint_path="./ckpt.t7", pre_train=None, data_root='./data'):
        self.learning_rate = learning_rate
        self.checkpoint_path = Tools.new_dir(checkpoint_path)
        self.resume = resume
        self.pre_train = pre_train
        self.data_root = data_root
        self.batch_size = batch_size

        self.low_dim = low_dim
        self.low_dim2 = low_dim2
        self.low_dim3 = low_dim3
        self.low_dim4 = low_dim4
        self.low_dim5 = low_dim5
        self.low_dim_list = [512, self.low_dim, self.low_dim2, self.low_dim3, self.low_dim4, self.low_dim5]
        self.ratio1 = ratio1
        self.ratio2 = ratio2
        self.ratio3 = ratio3
        self.ratio4 = ratio4
        self.ratio5 = ratio5

        self.t_epoch = t_epoch
        self.max_epoch = max_epoch
        self.first_epoch = first_epoch
        self.linear_bias = linear_bias
        self.has_l1 = has_l1
        self.l1_lambda = l1_lambda
        self.is_adjust_lambda = is_adjust_lambda
        self.is_loss_sum = is_loss_sum

        self.best_acc = 0

        self.train_set, self.train_loader, self.test_set, self.test_loader, self.class_name = CIFAR10Instance.data(
            self.data_root, batch_size=self.batch_size)

        self.net = HCResNet(HCBasicBlock, [2, 2, 2, 2], self.low_dim, self.low_dim2,
                            self.low_dim3, self.low_dim4, self.low_dim5, linear_bias=linear_bias).cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

        self._load_model(self.net)
        pass

    def _load_model(self, net):
        # Load checkpoint.
        if self.resume:
            Tools.print('==> Resuming from checkpoint {} ..'.format(self.checkpoint_path))
            checkpoint = torch.load(self.checkpoint_path)
            net.load_state_dict(checkpoint['net'])
            self.best_acc = checkpoint['acc']
            best_epoch = checkpoint['epoch']
            Tools.print("{} {}".format(self.best_acc, best_epoch))
            pass
        pass

    def obtain_label(self, result_file, is_train=False):
        self.net.eval()

        n_sample = self.train_set.__len__() if is_train else self.test_set.__len__()
        data_loader = self.train_loader if is_train else self.test_loader

        produce_class = ProduceClass(n_sample=n_sample, low_dim=self.low_dim, ratio=self.low_dim)
        produce_class2 = ProduceClass(n_sample=n_sample, low_dim=self.low_dim2, ratio=self.low_dim2)
        produce_class3 = ProduceClass(n_sample=n_sample, low_dim=self.low_dim3, ratio=self.low_dim3)
        produce_class4 = ProduceClass(n_sample=n_sample, low_dim=self.low_dim4, ratio=self.low_dim4)
        produce_class5 = ProduceClass(n_sample=n_sample, low_dim=self.low_dim5, ratio=self.low_dim5)

        produce_class.reset()
        produce_class2.reset()
        produce_class3.reset()
        produce_class4.reset()
        produce_class5.reset()

        target_class = np.zeros(shape=(n_sample, ), dtype=np.int)
        for batch_idx, (inputs, targets, indexes) in enumerate(data_loader):
            inputs, indexes = inputs.cuda(), indexes.cuda()
            feature_dict = self.net(inputs)
            produce_class.cal_label(feature_dict[FeatureName.L2norm1], indexes)
            produce_class2.cal_label(feature_dict[FeatureName.L2norm2], indexes)
            produce_class3.cal_label(feature_dict[FeatureName.L2norm3], indexes)
            produce_class4.cal_label(feature_dict[FeatureName.L2norm4], indexes)
            produce_class5.cal_label(feature_dict[FeatureName.L2norm5], indexes)
            target_class[indexes] = targets
            pass

        Tools.print("Epoch: 1-{}/{} 2-{}/{} 3-{}/{} 4-{}/{} 5-{}/{}".format(
            produce_class.count, produce_class.count_2,
            produce_class2.count, produce_class2.count_2,
            produce_class3.count, produce_class3.count_2,
            produce_class4.count, produce_class4.count_2,
            produce_class5.count, produce_class5.count_2))

        result = {"0": target_class,
                  "1": produce_class.classes.data.obj,
                  "2": produce_class2.classes.data.obj,
                  "3": produce_class3.classes.data.obj,
                  "4": produce_class4.classes.data.obj,
                  "5": produce_class5.classes.data.obj}

        Tools.write_to_pkl(result_file, result)
        pass

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    """
    # 11_class_1024_5level_512_256_128_64_no_1600_32_1_l1_sum_0_54321
    88.38(1024) 88.33(512) 88.17(256) 88.14(128) 88.50(64)
    """

    _start_epoch = 0
    _max_epoch = 1600
    _learning_rate = 0.01
    _first_epoch, _t_epoch = 200, 100
    _low_dim, _low_dim2, _low_dim3, _low_dim4, _low_dim5 = 1024, 512, 256, 128, 64
    _ratio1, _ratio2, _ratio3, _ratio4, _ratio5 = 5, 4, 3, 2, 1
    _l1_lambda = 0.0
    _is_adjust_lambda = False

    _batch_size = 32
    _is_loss_sum = True
    _has_l1 = True
    _linear_bias = False
    _resume = True
    _pre_train = None
    # _pre_train = "./checkpoint/11_class_1024_5level_512_256_128_64_1600_no_32_1_l1_sum_0_54321/ckpt.t7"
    _name = "11_class_{}_5level_{}_{}_{}_{}_no_{}_{}_{}_l1_sum_{}_{}{}{}{}{}".format(
        _low_dim, _low_dim2, _low_dim3, _low_dim4, _low_dim5, _max_epoch, _batch_size,
        0 if _linear_bias else 1, 1 if _is_adjust_lambda else 0, _ratio1, _ratio2, _ratio3, _ratio4, _ratio5)
    _checkpoint_path = "./checkpoint/{}/ckpt.t7".format(_name)

    Tools.print()
    Tools.print("name={}".format(_name))
    Tools.print("low_dim={} low_dim2={} low_dim3={} low_dim4={} low_dim5={}".format(
        _low_dim, _low_dim2, _low_dim3, _low_dim4, _low_dim5))
    Tools.print("ratio1={} ratio2={} ratio3={} ratio4={} ratio5={}".format(_ratio1, _ratio2, _ratio3, _ratio4, _ratio5))
    Tools.print("learning_rate={} batch_size={}".format(_learning_rate, _batch_size))
    Tools.print("has_l1={} l1_lambda={} is_adjust_lambda={}".format(_has_l1, _l1_lambda, _is_adjust_lambda))
    Tools.print("pre_train={} checkpoint_path={}".format(_pre_train, _checkpoint_path))
    Tools.print()

    runner = HCRunner(low_dim=_low_dim, low_dim2=_low_dim2, low_dim3=_low_dim3, low_dim4=_low_dim4, low_dim5=_low_dim5,
                      ratio1=_ratio1, ratio2=_ratio2, ratio3=_ratio3, ratio4=_ratio4, ratio5=_ratio5,
                      linear_bias=_linear_bias, has_l1=_has_l1,
                      l1_lambda=_l1_lambda, is_adjust_lambda=_is_adjust_lambda,
                      is_loss_sum=_is_loss_sum, batch_size=_batch_size, learning_rate=_learning_rate,
                      max_epoch=_max_epoch, t_epoch=_t_epoch, first_epoch=_first_epoch,
                      resume=_resume, pre_train=_pre_train, checkpoint_path=_checkpoint_path)
    Tools.print()
    runner.obtain_label(result_file="./checkpoint/{}/class_clustering.pkl".format(_name))
    pass
