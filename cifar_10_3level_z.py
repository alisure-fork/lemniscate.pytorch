import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from alisuretool.Tools import Tools
from cifar_10_tool import HCBasicBlock, AverageMeter, HCLoss, KNN
from cifar_10_tool import ProduceClass, FeatureName, Normalize, CIFAR10Instance


class HCResNet(nn.Module):

    def __init__(self, block, num_blocks, low_dim=512, low_dim2=128,
                 low_dim3=10, linear_bias=True, is_vis=False):
        super(HCResNet, self).__init__()
        self.in_planes = 64
        self.is_vis = is_vis

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear_512 = nn.Linear(512 * block.expansion, low_dim, bias=linear_bias)
        self.linear_128 = nn.Linear(low_dim, low_dim2, bias=linear_bias)
        self.linear_10 = nn.Linear(low_dim2, low_dim3, bias=linear_bias)
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

        out_logits = self.linear_512(avgPool)
        out_l2norm = self.l2norm(out_logits)

        out_logits2 = self.linear_128(out_logits)
        out_l2norm2 = self.l2norm(out_logits2)

        out_logits3 = self.linear_10(out_logits2)
        out_l2norm3 = self.l2norm(out_logits3)

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

        return feature_dict

    pass


class HCRunner(object):

    def __init__(self, low_dim=512, low_dim2=128, low_dim3=10,
                 ratio1=3, ratio2=2, ratio3=1, batch_size=128,
                 is_loss_sum=False, is_adjust_lambda=False, l1_lambda=0.01, learning_rate=0.03,
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
        self.low_dim_list = [512, self.low_dim, self.low_dim2, self.low_dim3]
        self.ratio1 = ratio1
        self.ratio2 = ratio2
        self.ratio3 = ratio3

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
        self.train_num = self.train_set.__len__()

        self.net = HCResNet(HCBasicBlock, [2, 2, 2, 2], self.low_dim,
                            self.low_dim2, self.low_dim3, linear_bias=linear_bias).cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

        self._load_model(self.net)

        self.produce_class = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim, ratio=self.ratio1)
        self.produce_class2 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim2, ratio=self.ratio2)
        self.produce_class3 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim3, ratio=self.ratio3)
        self.criterion = HCLoss().cuda()  # define loss function
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        pass

    def _adjust_learning_rate(self, epoch):

        def _get_lr(_base_lr, now_epoch, _t_epoch=self.t_epoch, _eta_min=1e-05):
            return _eta_min + (_base_lr - _eta_min) * (1 + math.cos(math.pi * now_epoch / _t_epoch)) / 2

        t_epoch = self.t_epoch
        first_epoch = self.first_epoch
        if epoch < first_epoch + self.t_epoch * 0:  # 0-200
            learning_rate = self.learning_rate
        elif epoch < first_epoch + t_epoch * 1:  # 200-300
            learning_rate = self.learning_rate / 2
        elif epoch < first_epoch + t_epoch * 2:  # 300-400
            learning_rate = self.learning_rate / 4
        elif epoch < first_epoch + t_epoch * 3:  # 400-500
            learning_rate = _get_lr(self.learning_rate / 2.0, epoch - first_epoch - t_epoch * 2)
        elif epoch < first_epoch + t_epoch * 4:  # 500-600
            learning_rate = _get_lr(self.learning_rate / 2.0, epoch - first_epoch - t_epoch * 3)
        elif epoch < first_epoch + t_epoch * 5:  # 600-700
            learning_rate = _get_lr(self.learning_rate / 4.0, epoch - first_epoch - t_epoch * 4)
        elif epoch < first_epoch + t_epoch * 6:  # 700-800
            learning_rate = _get_lr(self.learning_rate / 4.0, epoch - first_epoch - t_epoch * 5)
        elif epoch < first_epoch + t_epoch * 7:  # 800-900
            learning_rate = _get_lr(self.learning_rate / 8.0, epoch - first_epoch - t_epoch * 6)
        elif epoch < first_epoch + t_epoch * 8:  # 900-1000
            learning_rate = _get_lr(self.learning_rate / 8.0, epoch - first_epoch - t_epoch * 7)
        elif epoch < first_epoch + t_epoch * 9:  # 1000-1100
            learning_rate = _get_lr(self.learning_rate / 16., epoch - first_epoch - t_epoch * 8)
        elif epoch < first_epoch + t_epoch * 10:  # 1100-1200
            learning_rate = _get_lr(self.learning_rate / 16., epoch - first_epoch - t_epoch * 9)
        elif epoch < first_epoch + t_epoch * 11:  # 1200-1300
            learning_rate = _get_lr(self.learning_rate / 32., epoch - first_epoch - t_epoch * 10)
        elif epoch < first_epoch + t_epoch * 12:  # 1300-1400
            learning_rate = _get_lr(self.learning_rate / 32., epoch - first_epoch - t_epoch * 11)
        elif epoch < first_epoch + t_epoch * 13:  # 1400-1500
            learning_rate = _get_lr(self.learning_rate / 64., epoch - first_epoch - t_epoch * 12)
        else:  # 1500-1600
            learning_rate = _get_lr(self.learning_rate / 64., epoch - first_epoch - t_epoch * 13)
            pass

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
            pass

        return learning_rate

    def _adjust_l1_lambda(self, epoch):
        if not self.is_adjust_lambda:
            return self.l1_lambda

        t_epoch = self.t_epoch
        first_epoch = self.first_epoch
        if epoch < first_epoch + self.t_epoch * 0:  # 0-200
            l1_lambda = 0.0
        elif epoch < first_epoch + t_epoch * 2:  # 200-400
            l1_lambda = 0.0
        elif epoch < first_epoch + t_epoch * 4:  # 400-600
            l1_lambda = 0.0
        elif epoch < first_epoch + t_epoch * 6:  # 600-800
            l1_lambda = 0.0
        elif epoch < first_epoch + t_epoch * 8:  # 800-1000
            l1_lambda = 0.0
        elif epoch < first_epoch + t_epoch * 10:  # 1000-1200
            l1_lambda = self.l1_lambda
        elif epoch < first_epoch + t_epoch * 12:  # 1200-1400
            l1_lambda = self.l1_lambda
        else:  # 1400-1600
            l1_lambda = self.l1_lambda
            pass
        return l1_lambda

    def _load_model(self, net):
        # Load PreTrain
        if self.pre_train:
            Tools.print('==> Pre train from checkpoint {} ..'.format(self.pre_train))
            checkpoint = torch.load(self.pre_train)
            net.load_state_dict(checkpoint['net'], strict=False)
            self.best_acc = checkpoint['acc']
            best_epoch = checkpoint['epoch']
            Tools.print("{} {}".format(self.best_acc, best_epoch))
            pass

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

    def test(self, epoch=0, t=0.1, loader_n=1):
        _acc = KNN.knn(epoch, self.net, self.low_dim_list,
                       self.train_loader, self.test_loader, 200, t, loader_n=loader_n)
        return _acc

    def _train_one_epoch(self, epoch, update_epoch=3):

        # Update
        try:
            if epoch % update_epoch == 0:
                self.net.eval()
                Tools.print("Update label {} .......".format(epoch))
                self.produce_class.reset()
                self.produce_class2.reset()
                self.produce_class3.reset()
                for batch_idx, (inputs, _, indexes) in enumerate(self.train_loader):
                    inputs, indexes = inputs.cuda(), indexes.cuda()
                    feature_dict = self.net(inputs)
                    self.produce_class.cal_label(feature_dict[FeatureName.L2norm1], indexes)
                    self.produce_class2.cal_label(feature_dict[FeatureName.L2norm2], indexes)
                    self.produce_class3.cal_label(feature_dict[FeatureName.L2norm3], indexes)
                    pass
                Tools.print("Epoch: [{}] 1-{}/{} 2-{}/{} 3-{}/{}".format(
                    epoch, self.produce_class.count, self.produce_class.count_2,
                    self.produce_class2.count, self.produce_class2.count_2,
                    self.produce_class3.count, self.produce_class3.count_2))
                pass
        finally:
            pass

        # Train
        try:
            self.net.train()
            _learning_rate_ = self._adjust_learning_rate(epoch)
            _l1_lambda_ = self._adjust_l1_lambda(epoch)
            Tools.print('Epoch: {} lr={} lambda={}'.format(epoch, _learning_rate_, _l1_lambda_))

            avg_loss_1, avg_loss_1_1, avg_loss_1_2 = AverageMeter(), AverageMeter(), AverageMeter()
            avg_loss_2, avg_loss_2_1, avg_loss_2_2 = AverageMeter(), AverageMeter(), AverageMeter()
            avg_loss_3, avg_loss_3_1, avg_loss_3_2 = AverageMeter(), AverageMeter(), AverageMeter()

            for batch_idx, (inputs, _, indexes) in enumerate(self.train_loader):
                inputs, indexes = inputs.cuda(), indexes.cuda()
                self.optimizer.zero_grad()

                feature_dict = self.net(inputs)

                targets = self.produce_class.get_label(indexes)
                targets2 = self.produce_class2.get_label(indexes)
                targets3 = self.produce_class3.get_label(indexes)

                params = [_ for _ in self.net.module.parameters()]
                loss_1, loss_1_1, loss_1_2 = self.criterion(
                    feature_dict[FeatureName.Logits1], targets, params[-3], _l1_lambda_)
                loss_2, loss_2_1, loss_2_2 = self.criterion(
                    feature_dict[FeatureName.Logits2], targets2, params[-2], _l1_lambda_)
                loss_3, loss_3_1, loss_3_2 = self.criterion(
                    feature_dict[FeatureName.Logits3], targets3, params[-1], _l1_lambda_)

                avg_loss_1.update(loss_1.item(), inputs.size(0))
                avg_loss_1_1.update(loss_1_1.item(), inputs.size(0))
                avg_loss_1_2.update(loss_1_2.item(), inputs.size(0))
                avg_loss_2.update(loss_2.item(), inputs.size(0))
                avg_loss_2_1.update(loss_2_1.item(), inputs.size(0))
                avg_loss_2_2.update(loss_2_2.item(), inputs.size(0))
                avg_loss_3.update(loss_3.item(), inputs.size(0))
                avg_loss_3_1.update(loss_3_1.item(), inputs.size(0))
                avg_loss_3_2.update(loss_3_2.item(), inputs.size(0))

                loss = loss_1 + loss_2 + loss_3 if self.has_l1 else loss_1_1 + loss_2_1 + loss_3_1

                loss.backward()

                self.optimizer.step()
                pass

            Tools.print(
                'Epoch: {}/{} Loss 1: {:.4f}({:.4f}/{:.4f}) '
                'Loss 2: {:.4f}({:.4f}/{:.4f}) Loss 3: {:.4f}({:.4f}/{:.4f})'.format(
                    epoch, len(self.train_loader), avg_loss_1.avg, avg_loss_1_1.avg, avg_loss_1_2.avg,avg_loss_2.avg,
                    avg_loss_2_1.avg, avg_loss_2_2.avg, avg_loss_3.avg, avg_loss_3_1.avg, avg_loss_3_2.avg))
        finally:
            pass

        # Test
        try:
            Tools.print("Test [{}] .......".format(epoch))
            _acc = self.test(epoch=epoch)
            if _acc > self.best_acc:
                Tools.print('Saving..')
                state = {'net': self.net.state_dict(), 'acc': _acc, 'epoch': epoch}
                torch.save(state, self.checkpoint_path)
                self.best_acc = _acc
                pass
            Tools.print('Epoch: [{}] best accuracy: {:.2f}'.format(epoch, self.best_acc * 100))
        finally:
            pass

        pass

    def train(self, start_epoch, update_epoch=3):
        for epoch in range(start_epoch, self.max_epoch):
            Tools.print()
            self._train_one_epoch(epoch, update_epoch=update_epoch)
            pass
        pass

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    """
    # 11_class_1024_3level_256_64_1000_no_memory_1_l1_sum
    85.93(1024, 13462/1716) 85.99(256, 13462/1716) 86.20(64, 12505/2035)

    # 11_class_1024_3level_256_64_1600_no_32_1_l1_sum_0_333
    88.20(1024, 15770/2614) 88.21(256, 10121/769) 86.61 (64, 7463/1141)

    # 11_class_1024_3level_256_64_1600_no_32_1_l1_sum_0_555 1380
    88.01(1024, x) 88.07(256, x) 83.61 (64, x)
    
    # 11_class_1024_3level_256_64_1600_no_32_1_l1_sum_1_321
    87.25(1024) 87.19(256) 87.24(64)
    
    # 11_class_1024_3level_256_64_1600_no_128_1_l1_sum_1_321
    84.75(1024) 84.88(256) 84.74(64)


    # 11_class_1024_3level_256_64_1600_no_32_1_l1_sum_0_321
    87.64(1024, 17809/3045) 87.65(256, 13821/1756) 87.59(64, 12949/1997)
    # 11_class_1024_3level_256_64_1600_no_32_1_l1_sum_0_321
    88.41(1024, 16684/2523) 88.63(256, 12625/1250) 88.36(64, 11818/1902)
    # 11_class_1024_3level_256_64_1600_no_32_1_l1_sum_0_321
    88.60(512) 88.47(1024) 88.62(256) 88.45(64)

    """

    """
    10_aa_update_epoch_class_1024_3level_256_64_1600_no_32_1_l1_sum_0_321_3
    2019-11-13 06:42:40 has_l1=True l1_lambda=0.0 is_adjust_lambda=False
    2019-11-13 06:42:40 Epoch: 1496 lambda=0.0
    2019-11-13 06:43:47 Test 1496 0 Top1=87.43 Top5=99.49
    2019-11-13 06:43:47 Test 1496 1 Top1=87.44 Top5=99.46
    2019-11-13 06:43:47 Test 1496 2 Top1=87.50 Top5=99.46
    2019-11-13 06:43:47 Test 1496 3 Top1=87.55 Top5=99.40

    10_aa_update_epoch_class_1024_3level_256_64_1600_no_32_1_l1_sum_0_321_5
    2019-11-13 02:34:22 has_l1=True l1_lambda=0.0 is_adjust_lambda=False
    2019-11-13 02:34:22 Epoch: 1299 lambda=0.0
    2019-11-13 02:35:30 Test 1299 0 Top1=86.78 Top5=99.52
    2019-11-13 02:35:30 Test 1299 1 Top1=86.60 Top5=99.44
    2019-11-13 02:35:30 Test 1299 2 Top1=86.67 Top5=99.47
    2019-11-13 02:35:30 Test 1299 3 Top1=86.28 Top5=99.34

    10_aa_update_epoch_class_1024_3level_256_64_1600_no_32_1_l1_sum_0_321_1
    2019-11-13 10:15:13 has_l1=True l1_lambda=0.0 is_adjust_lambda=False
    2019-11-15 01:21:43 Test 1589 0 Top1=88.59 Top5=99.58
    2019-11-15 01:21:43 Test 1589 1 Top1=88.23 Top5=99.58
    2019-11-15 01:21:43 Test 1589 2 Top1=88.28 Top5=99.57
    2019-11-15 01:21:43 Test 1589 3 Top1=88.06 Top5=99.51

    10_a_l1_class_1024_3level_256_64_1600_no_32_1_l1_sum_1_321
    2019-11-11 23:41:16 has_l1=True l1_lambda=0.001 is_adjust_lambda=True
    2019-11-13 13:39:24 Test 1491 0 Top1=88.15 Top5=99.58
    2019-11-13 13:39:24 Test 1491 1 Top1=87.80 Top5=99.53
    2019-11-13 13:39:24 Test 1491 2 Top1=87.67 Top5=99.61
    2019-11-13 13:39:24 Test 1491 3 Top1=87.69 Top5=99.55
    
    10_a_l1_class_1024_3level_256_64_1600_no_32_1_l1_sum_0_321
    2019-11-11 23:41:39 has_l1=True l1_lambda=0.001 is_adjust_lambda=False
    2019-11-13 14:05:02 Test 1482 0 Top1=88.27 Top5=99.56
    2019-11-13 14:05:02 Test 1482 1 Top1=87.77 Top5=99.51
    2019-11-13 14:05:02 Test 1482 2 Top1=87.89 Top5=99.51
    2019-11-13 14:05:02 Test 1482 3 Top1=87.72 Top5=99.49
    
    10_a_l1_class_1024_3level_256_64_1600_no_32_1_l1_sum_0_321
    2019-11-13 18:50:34 has_l1=True l1_lambda=0.05 is_adjust_lambda=False
    2019-11-15 07:02:25 Test 1463 0 Top1=88.28 Top5=99.56
    2019-11-15 07:02:25 Test 1463 1 Top1=87.82 Top5=99.52
    2019-11-15 07:02:25 Test 1463 2 Top1=87.60 Top5=99.49
    2019-11-15 07:02:25 Test 1463 3 Top1=87.65 Top5=99.48
    
    10_a_l1_class_1024_3level_256_64_1600_no_32_1_l1_sum_0_321
    2019-11-13 18:50:05 has_l1=True l1_lambda=1.0 is_adjust_lambda=False
    2019-11-15 08:52:16 Test 1499 0 Top1=85.51 Top5=99.31
    2019-11-15 08:52:16 Test 1499 1 Top1=84.78 Top5=99.17
    2019-11-15 08:52:16 Test 1499 2 Top1=82.90 Top5=98.72
    2019-11-15 08:52:16 Test 1499 3 Top1=83.81 Top5=98.98
    """
    _start_epoch = 0
    _max_epoch = 1600
    _learning_rate = 0.01
    _first_epoch, _t_epoch = 200, 100
    _low_dim, _low_dim2, _low_dim3 = 1024, 256, 64
    _ratio1, _ratio2, _ratio3 = 3, 2, 1
    _l1_lambda = 0.0
    _is_adjust_lambda = False
    _update_epoch = 1

    _batch_size = 32
    _is_loss_sum = True
    _has_l1 = True
    _linear_bias = False
    _resume = False
    _pre_train = None
    # _pre_train = "./checkpoint/11_class_1024_3level_256_64_1600_no_32_1_l1_sum_0_321/ckpt.t7"
    _name = "10_aa_update_epoch_class_{}_3level_{}_{}_{}_no_{}_{}_l1_sum_{}_{}{}{}_{}".format(
        _low_dim, _low_dim2, _low_dim3, _max_epoch, _batch_size,
        0 if _linear_bias else 1, 1 if _is_adjust_lambda else 0, _ratio1, _ratio2, _ratio3, _update_epoch)
    _checkpoint_path = "./checkpoint/{}/ckpt.t7".format(_name)

    Tools.print()
    Tools.print("name={}".format(_name))
    Tools.print("low_dim={} low_dim2={} low_dim3={}".format(_low_dim, _low_dim2, _low_dim3))
    Tools.print("ratio1={} ratio2={} ratio3={}".format(_ratio1, _ratio2, _ratio3))
    Tools.print("learning_rate={} batch_size={}".format(_learning_rate, _batch_size))
    Tools.print("has_l1={} l1_lambda={} is_adjust_lambda={}".format(_has_l1, _l1_lambda, _is_adjust_lambda))
    Tools.print("pre_train={} checkpoint_path={}".format(_pre_train, _checkpoint_path))
    Tools.print()

    runner = HCRunner(low_dim=_low_dim, low_dim2=_low_dim2, low_dim3=_low_dim3,
                      ratio1=_ratio1, ratio2=_ratio2, ratio3=_ratio3,
                      linear_bias=_linear_bias, has_l1=_has_l1,
                      l1_lambda=_l1_lambda, is_adjust_lambda=_is_adjust_lambda,
                      is_loss_sum=_is_loss_sum, batch_size=_batch_size, learning_rate=_learning_rate,
                      max_epoch=_max_epoch, t_epoch=_t_epoch, first_epoch=_first_epoch,
                      resume=_resume, pre_train=_pre_train, checkpoint_path=_checkpoint_path)
    Tools.print()
    acc = runner.test()
    Tools.print('Random accuracy: {:.2f}'.format(acc * 100))
    runner.train(start_epoch=_start_epoch, update_epoch=_update_epoch)
    Tools.print()
    acc = runner.test(loader_n=2)
    Tools.print('final accuracy: {:.2f}'.format(acc * 100))
    pass
