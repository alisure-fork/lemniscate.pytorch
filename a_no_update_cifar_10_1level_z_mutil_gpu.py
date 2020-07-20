import os
import math
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
from cifar_10_tool import HCBasicBlock, AverageMeter, HCLoss, KNN
from cifar_10_tool import ProduceClass, FeatureName, Normalize, CIFAR10Instance


class HCResNet(nn.Module):

    def __init__(self, block, num_blocks, low_dim=512, linear_bias=True, is_vis=False):
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

        feature_dict = {}
        if self.is_vis:
            feature_dict[FeatureName.x] = x
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


class HCRunner(object):

    def __init__(self, low_dim=512, ratio=1, batch_size=128, num_workers=16,
                 is_adjust_lambda=False, l1_lambda=0.1, learning_rate=0.03,
                 linear_bias=True, has_l1=False, max_epoch=1000, t_epoch=300, first_epoch=200,
                 resume=False, checkpoint_path="./ckpt.t7", pre_train=None, data_root='./data'):
        self.learning_rate = learning_rate
        self.checkpoint_path = Tools.new_dir(checkpoint_path)
        self.resume = resume
        self.pre_train = pre_train
        self.data_root = data_root
        self.batch_size = batch_size

        self.low_dim = low_dim
        self.low_dim_list = [512, self.low_dim]
        self.ratio = ratio

        self.t_epoch = t_epoch
        self.max_epoch = max_epoch
        self.first_epoch = first_epoch
        self.linear_bias = linear_bias
        self.has_l1 = has_l1
        self.l1_lambda = l1_lambda
        self.is_adjust_lambda = is_adjust_lambda

        self.best_acc = 0

        self.train_set, self.train_loader, self.test_set, self.test_loader, self.class_name = CIFAR10Instance.data(
            self.data_root, batch_size=self.batch_size, num_workers=num_workers)
        self.train_num = self.train_set.__len__()

        #############################################################
        self.net = HCResNet(HCBasicBlock, [2, 2, 2, 2], self.low_dim, linear_bias=linear_bias)
        self.net = nn.DataParallel(self.net)
        self.net = self.net.cuda()
        self._load_model(self.net)

        cudnn.benchmark = True
        #############################################################

        self.produce_class1 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim, ratio=self.ratio)
        self.produce_class2 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim, ratio=self.ratio)
        self.produce_class1.init()
        self.produce_class2.init()

        self.criterion = HCLoss().cuda()  # define loss function
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        # self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        pass

    def _adjust_learning_rate2(self, epoch):

        t_epoch = self.t_epoch
        first_epoch = self.first_epoch
        if epoch < first_epoch + self.t_epoch * 0:  # 0-500
            learning_rate = self.learning_rate
        elif epoch < first_epoch + t_epoch * 1:  # 500-1000
            learning_rate = self.learning_rate / 10
        else:  # 1000-1500
            learning_rate = self.learning_rate / 100
            pass

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
            pass

        return learning_rate

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
            net.load_state_dict(checkpoint['net'], strict=True)
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
                       self.train_loader, self.test_loader, 200, t, loader_n=loader_n, temp_size=100)
        return _acc

    def _train_one_epoch(self, epoch, eval_epoch=10):
        # Train
        try:
            self.net.train()
            _learning_rate_ = self._adjust_learning_rate(epoch)
            _l1_lambda_ = self._adjust_l1_lambda(epoch)
            Tools.print('Epoch: [{}] lr={} lambda={}'.format(epoch, _learning_rate_, _l1_lambda_))

            avg_loss_1, avg_loss_1_1, avg_loss_1_2 = AverageMeter(), AverageMeter(), AverageMeter()

            self.produce_class1.reset()
            for batch_idx, (inputs, _, indexes) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                inputs, indexes = inputs.cuda(), indexes.cuda()
                self.optimizer.zero_grad()

                feature_dict = self.net(inputs)

                self.produce_class1.cal_label(feature_dict[FeatureName.L2norm1], indexes)
                targets = self.produce_class2.get_label(indexes)

                params = [_ for _ in self.net.module.parameters()]
                loss_1, loss_1_1, loss_1_2 = self.criterion(
                    feature_dict[FeatureName.Logits1], targets, params[-1], _l1_lambda_)

                avg_loss_1.update(loss_1.item(), inputs.size(0))
                avg_loss_1_1.update(loss_1_1.item(), inputs.size(0))
                avg_loss_1_2.update(loss_1_2.item(), inputs.size(0))

                loss = loss_1 if self.has_l1 else loss_1_1
                loss.backward()

                self.optimizer.step()
                pass

            classes = self.produce_class2.classes
            self.produce_class2.classes = self.produce_class1.classes
            self.produce_class1.classes = classes
            Tools.print("Train: [{}] 1-{}/{}".format(epoch, self.produce_class1.count, self.produce_class1.count_2))

            Tools.print('Train: [{}] {} Loss 1: {:.4f}({:.4f}/{:.4f})'.format(
                epoch, len(self.train_loader), avg_loss_1.avg, avg_loss_1_1.avg, avg_loss_1_2.avg))
        finally:
            pass

        # Test
        if epoch % eval_epoch == 0:
            Tools.print("Test:  [{}] .......".format(epoch))
            _acc = self.test(epoch=epoch)
            if _acc > self.best_acc:
                Tools.print('Saving..')
                state = {'net': self.net.state_dict(), 'acc': _acc, 'epoch': epoch}
                torch.save(state, self.checkpoint_path)
                self.best_acc = _acc
                pass
            Tools.print('Test:  [{}] best accuracy: {:.2f}'.format(epoch, self.best_acc))
            pass

        pass

    def _train_one_epoch2(self, epoch, update_epoch=3, eval_epoch=10):

        # Update
        try:
            if epoch % update_epoch == 0:
                self.net.eval()
                Tools.print("Update label {} .......".format(epoch))
                self.produce_class1.reset()
                with torch.no_grad():
                    for batch_idx, (inputs, _, indexes) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                        inputs, indexes = inputs.cuda(), indexes.cuda()
                        feature_dict = self.net(inputs)
                        self.produce_class1.cal_label(feature_dict[FeatureName.L2norm1], indexes)
                        pass
                    pass
                Tools.print("Epoch: [{}] 1-{}/{}".format(
                    epoch, self.produce_class1.count, self.produce_class1.count_2))
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

            for batch_idx, (inputs, _, indexes) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                inputs, indexes = inputs.cuda(), indexes.cuda()
                self.optimizer.zero_grad()

                feature_dict = self.net(inputs)

                targets = self.produce_class1.get_label(indexes)
                params = [_ for _ in self.net.module.parameters()]
                loss_1, loss_1_1, loss_1_2 = self.criterion(
                    feature_dict[FeatureName.Logits1], targets, params[-1], _l1_lambda_)

                avg_loss_1.update(loss_1.item(), inputs.size(0))
                avg_loss_1_1.update(loss_1_1.item(), inputs.size(0))
                avg_loss_1_2.update(loss_1_2.item(), inputs.size(0))

                loss = loss_1 if self.has_l1 else loss_1_1
                loss.backward()

                self.optimizer.step()
                pass

            Tools.print(
                'Epoch: {}/{} Loss 1: {:.4f}({:.4f}/{:.4f})'.format(
                    epoch, len(self.train_loader), avg_loss_1.avg, avg_loss_1_1.avg, avg_loss_1_2.avg))
        finally:
            pass

        # Test
        if epoch % eval_epoch == 0:
            Tools.print("Test [{}] .......".format(epoch))
            _acc = self.test(epoch=epoch)
            if _acc > self.best_acc:
                Tools.print('Saving..')
                state = {'net': self.net.state_dict(), 'acc': _acc, 'epoch': epoch}
                torch.save(state, self.checkpoint_path)
                self.best_acc = _acc
                pass
            Tools.print('Epoch: [{}] best accuracy: {:.2f}'.format(epoch, self.best_acc))
            pass

        pass

    def train(self, start_epoch):
        if start_epoch >= 0:
            self.net.eval()
            Tools.print("Update label {} .......".format(start_epoch))
            self.produce_class1.reset()
            with torch.no_grad():
                for batch_idx, (inputs, _, indexes) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                    inputs, indexes = inputs.cuda(), indexes.cuda()
                    feature_dict = self.net(inputs)
                    self.produce_class1.cal_label(feature_dict[FeatureName.L2norm1], indexes)
                    pass
            classes = self.produce_class2.classes
            self.produce_class2.classes = self.produce_class1.classes
            self.produce_class1.classes = classes
            Tools.print("Train: [{}] 1-{}/{}".format(start_epoch, self.produce_class1.count, self.produce_class1.count_2))
            pass

        for epoch in range(start_epoch, self.max_epoch + 1):
            Tools.print()
            # self._train_one_epoch(epoch, eval_epoch=10)
            self._train_one_epoch2(epoch, update_epoch=1)
            pass
        pass

    pass


"""
2020-05-23 11:31:42 Epoch: [1286] lr=2.4394908314514573e-05 lambda=0.0
2020-05-23 11:32:37 Train: [1286] 1-16616/2898
2020-05-23 11:32:37 Train: [1286] 1563 Loss 1: 1.1815(1.1815/21.9197)
2020-05-23 11:32:37 Test:  [1286] .......
2020-05-23 11:32:53 Test:  [1286] 0 Top1=85.10 Top5=99.47
2020-05-23 11:32:53 Test:  [1286] 1 Top1=84.41 Top5=99.37
2020-05-23 11:32:53 Saving..
2020-05-23 11:32:53 Test:  [1286] best accuracy: 84.41


2020-07-14 13:46:33 name=1level_128_1600_no_1024_1_l1_sum_0_1
2020-07-14 13:46:33 low_dim=128 ratio=1
2020-07-14 13:46:33 learning_rate=0.01 batch_size=1024
2020-07-14 13:46:33 has_l1=True l1_lambda=0.0 is_adjust_lambda=False
2020-07-14 13:46:33 pre_train=None checkpoint_path=./checkpoint/cifar10/1level_128_1600_no_1024_1_l1_sum_0_1/ckpt.t7
2020-07-14 19:53:51 Epoch: [1600] lr=1e-05 lambda=0.0
2020-07-14 19:54:35 Test:  [0] 0 Top1=73.67 Top5=98.25
2020-07-14 19:54:35 Test:  [0] 1 Top1=71.66 Top5=98.03
2020-07-14 19:54:42 Test:  [0] 0 Top1=71.17 Top5=98.44
2020-07-14 19:54:42 Test:  [0] 1 Top1=69.35 Top5=98.23
2020-07-14 19:54:42 final accuracy: 71.66


2020-07-15 09:39:41 name=1level_128_1500_no_3072_1_l1_sum_0_1
2020-07-15 09:39:41 low_dim=128 ratio=1
2020-07-15 09:39:41 learning_rate=0.01 batch_size=3072
2020-07-15 09:39:41 has_l1=True l1_lambda=0.0 is_adjust_lambda=False
2020-07-15 09:39:41 pre_train=None checkpoint_path=./checkpoint/cifar10/1level_128_1500_no_3072_1_l1_sum_0_1/ckpt.t7
2020-07-15 15:25:49 Train: [1500] 1-9973/2249
2020-07-15 15:26:21 Test:  [0] 0 Top1=77.72 Top5=98.65
2020-07-15 15:26:21 Test:  [0] 1 Top1=76.29 Top5=98.49
2020-07-15 15:26:28 Test:  [0] 0 Top1=79.63 Top5=99.37
2020-07-15 15:26:28 Test:  [0] 1 Top1=74.51 Top5=98.54
2020-07-15 15:26:28 final accuracy: 76.29
2020-07-16 21:46:13 final accuracy: 91.85/82.53


2020-07-15 15:34:06 name=1level_128_1500_no_64_1_l1_sum_0_1
2020-07-15 15:34:06 low_dim=128 ratio=1
2020-07-15 15:34:06 learning_rate=0.01 batch_size=64
2020-07-15 15:34:06 has_l1=True l1_lambda=0.0 is_adjust_lambda=False
2020-07-15 15:34:06 pre_train=None checkpoint_path=./checkpoint/cifar10/1level_128_1500_no_64_1_l1_sum_0_1/ckpt.t7
2020-07-16 16:47:39 Epoch: [1500] lr=0.0001 lambda=0.0
2020-07-16 16:48:38 Train: [1500] 1-16339/2872
2020-07-16 16:48:38 Train: [1500] 782 Loss 1: 1.1789(1.1789/22.9961)
2020-07-16 16:48:38 Test:  [1500] .......
2020-07-16 16:49:02 Test:  [1500] 0 Top1=84.23 Top5=99.36
2020-07-16 16:49:02 Test:  [1500] 1 Top1=83.49 Top5=99.24
2020-07-16 16:49:02 Test:  [1500] best accuracy: 83.69
2020-07-16 16:49:26 Test:  [0] 0 Top1=84.23 Top5=99.36
2020-07-16 16:49:26 Test:  [0] 1 Top1=83.49 Top5=99.24
2020-07-16 16:49:56 Test:  [0] 0 Top1=82.35 Top5=99.38
2020-07-16 16:49:56 Test:  [0] 1 Top1=81.50 Top5=99.31
2020-07-16 16:49:56 final accuracy: 83.49


2020-07-15 15:32:23 name=1level_128_1500_no_64_1_l1_sum_0_1
2020-07-15 15:32:23 low_dim=128 ratio=1
2020-07-15 15:32:23 learning_rate=0.01 batch_size=64
2020-07-15 15:32:23 has_l1=True l1_lambda=0.0 is_adjust_lambda=False
2020-07-15 15:32:23 pre_train=None checkpoint_path=./checkpoint/cifar10/1level_128_1500_no_64_1_l1_sum_0_1/ckpt.t7
2020-07-16 17:22:47 Epoch: [1500] lr=0.0001 lambda=0.0
2020-07-16 17:23:42 Train: [1500] 1-21080/2645
2020-07-16 17:23:42 Train: [1500] 782 Loss 1: 1.4593(1.4593/311.0285)
2020-07-16 17:23:42 Test:  [1500] .......
2020-07-16 17:24:06 Test:  [1500] 0 Top1=75.47 Top5=98.38
2020-07-16 17:24:06 Test:  [1500] 1 Top1=75.35 Top5=98.38
2020-07-16 17:24:06 Test:  [1500] best accuracy: 75.54
2020-07-16 17:24:29 Test:  [0] 0 Top1=75.47 Top5=98.38
2020-07-16 17:24:29 Test:  [0] 1 Top1=75.35 Top5=98.38
2020-07-16 17:24:57 Test:  [0] 0 Top1=73.00 Top5=98.07
2020-07-16 17:24:57 Test:  [0] 1 Top1=72.74 Top5=97.99
2020-07-16 17:24:57 final accuracy: 75.35
2020-07-16 21:49:34 final accuracy: 75.68/76.96


2020-07-16 22:33:19 name=1level_128_1600_no_256_1_l1_sum_0_1
2020-07-16 22:33:19 low_dim=128 ratio=1
2020-07-16 22:33:19 learning_rate=0.01 batch_size=256
2020-07-16 22:33:19 has_l1=True l1_lambda=0.0 is_adjust_lambda=False
2020-07-16 22:33:19 pre_train=None checkpoint_path=./checkpoint/cifar10/1level_128_1600_no_256_1_l1_sum_0_1/ckpt.t7
2020-07-17 06:32:06 Test:  [0] 0 Top1=81.37 Top5=99.20
2020-07-17 06:32:06 Test:  [0] 1 Top1=79.93 Top5=99.03
2020-07-17 06:32:15 Test:  [0] 0 Top1=79.27 Top5=99.35
2020-07-17 06:32:15 Test:  [0] 1 Top1=78.61 Top5=99.28
2020-07-17 06:32:15 final accuracy: 79.93

2020-07-18 15:41:37 Update label 1600 .......
2020-07-18 15:41:53 Epoch: [1600] 1-11665/2005
2020-07-18 15:41:53 Epoch: 1600 lr=1e-05 lambda=0.0
2020-07-18 15:42:33 Epoch: 1600/1563 Loss 1: 0.9300(0.9300/21.6776)
2020-07-18 15:42:33 Test [1600] .......
2020-07-18 15:42:44 Test:  [1600] 0 Top1=85.38 Top5=99.35
2020-07-18 15:42:44 Test:  [1600] 1 Top1=85.22 Top5=99.39
2020-07-18 15:42:44 Epoch: [1600] best accuracy: 8542.00
2020-07-18 15:42:55 Test:  [0] 0 Top1=85.38 Top5=99.35
2020-07-18 15:42:55 Test:  [0] 1 Top1=85.22 Top5=99.39
2020-07-18 15:43:13 Test:  [0] 0 Top1=83.76 Top5=99.58
2020-07-18 15:43:13 Test:  [0] 1 Top1=83.64 Top5=99.59
2020-07-18 15:43:13 final accuracy: 85.22

2020-07-18 05:02:03 Epoch: [1600] lr=1e-05 lambda=0.0
2020-07-18 05:02:45 Train: [1600] 1-15119/3112
2020-07-18 05:02:45 Train: [1600] 1563 Loss 1: 1.0668(1.0668/21.9402)
2020-07-18 05:02:45 Test:  [1600] .......
2020-07-18 05:02:57 Test:  [1600] 0 Top1=84.56 Top5=99.41
2020-07-18 05:02:57 Test:  [1600] 1 Top1=83.81 Top5=99.33
2020-07-18 05:02:57 Test:  [1600] best accuracy: 84.08
2020-07-18 05:03:08 Test:  [0] 0 Top1=84.56 Top5=99.41
2020-07-18 05:03:08 Test:  [0] 1 Top1=83.81 Top5=99.33
2020-07-18 05:03:25 Test:  [0] 0 Top1=82.65 Top5=99.46
2020-07-18 05:03:25 Test:  [0] 1 Top1=82.25 Top5=99.41
2020-07-18 05:03:25 final accuracy: 83.81
"""


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    _start_epoch = 0
    _max_epoch = 1600
    _learning_rate = 0.01
    # _first_epoch, _t_epoch = 1300, 100
    # _first_epoch, _t_epoch = 500, 500
    _first_epoch, _t_epoch = 200, 100
    _low_dim = 128
    _ratio = 1
    _l1_lambda = 0.0
    _is_adjust_lambda = False

    # _batch_size = 32 * 3 * 8 * 4
    _batch_size = 64 * 1
    # _batch_size = 32 * 1
    # _batch_size = 12 * 1
    _has_l1 = True
    _linear_bias = False
    _data_root = "/mnt/4T/Data/data/CIFAR"
    _resume = False
    _pre_train = None
    # _pre_train = "./checkpoint/11_class_128_1level_1600_no_32_1_l1_sum_0_1/ckpt.t7"
    _name = "1level_{}_{}_no_{}_{}_l1_sum_{}_{}2".format(_low_dim, _max_epoch, _batch_size, 0 if _linear_bias else 1,
                                                         1 if _is_adjust_lambda else 0, _ratio)
    _checkpoint_path = "./checkpoint/cifar10/{}/ckpt.t7".format(_name)

    Tools.print()
    Tools.print("name={}".format(_name))
    Tools.print("low_dim={} ratio={}".format(_low_dim, _ratio))
    Tools.print("learning_rate={} batch_size={}".format(_learning_rate, _batch_size))
    Tools.print("has_l1={} l1_lambda={} is_adjust_lambda={}".format(_has_l1, _l1_lambda, _is_adjust_lambda))
    Tools.print("pre_train={} checkpoint_path={}".format(_pre_train, _checkpoint_path))
    Tools.print()

    runner = HCRunner(low_dim=_low_dim, ratio=_ratio, data_root=_data_root,
                      linear_bias=_linear_bias, has_l1=_has_l1, num_workers=16,
                      l1_lambda=_l1_lambda, is_adjust_lambda=_is_adjust_lambda,
                      batch_size=_batch_size, learning_rate=_learning_rate,
                      max_epoch=_max_epoch, t_epoch=_t_epoch, first_epoch=_first_epoch,
                      resume=_resume, pre_train=_pre_train, checkpoint_path=_checkpoint_path)
    Tools.print()
    acc = runner.test()
    Tools.print('Random accuracy: {:.2f}'.format(acc))
    runner.train(start_epoch=_start_epoch)
    Tools.print()
    acc = runner.test(loader_n=2)
    Tools.print('final accuracy: {:.2f}'.format(acc))
    pass
