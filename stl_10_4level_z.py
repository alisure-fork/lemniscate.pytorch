import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from alisuretool.Tools import Tools
from cifar_10_tool import HCBasicBlock, AverageMeter, HCLoss, KNN
from cifar_10_tool import ProduceClass, FeatureName, Normalize, STL10Instance


class HCResNet(nn.Module):

    def __init__(self, block, num_blocks, low_dim=512, low_dim2=128,
                 low_dim3=10, low_dim4=10, linear_bias=True, input_size=32, conv1_stride=1, is_vis=False):
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
        self.linear_512 = nn.Linear(low_dim, low_dim2, bias=linear_bias)
        self.linear_256 = nn.Linear(low_dim2, low_dim3, bias=linear_bias)
        self.linear_128 = nn.Linear(low_dim3, low_dim4, bias=linear_bias)
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

        out_logits2 = self.linear_512(out_logits)
        out_l2norm2 = self.l2norm(out_logits2)

        out_logits3 = self.linear_256(out_logits2)
        out_l2norm3 = self.l2norm(out_logits3)

        out_logits4 = self.linear_128(out_logits3)
        out_l2norm4 = self.l2norm(out_logits4)

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

        return feature_dict

    pass


class HCRunner(object):

    def __init__(self, low_dim=512, low_dim2=128, low_dim3=10, low_dim4=10,
                 ratio1=3, ratio2=2, ratio3=1, ratio4=1, batch_size=128, input_size=32, conv1_stride=1,
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
        self.low_dim_list = [512, self.low_dim, self.low_dim2, self.low_dim3, self.low_dim4]
        self.ratio1 = ratio1
        self.ratio2 = ratio2
        self.ratio3 = ratio3
        self.ratio4 = ratio4
        self.input_size = input_size
        self.conv1_stride = conv1_stride

        self.t_epoch = t_epoch
        self.max_epoch = max_epoch
        self.first_epoch = first_epoch
        self.linear_bias = linear_bias
        self.has_l1 = has_l1
        self.l1_lambda = l1_lambda
        self.is_adjust_lambda = is_adjust_lambda
        self.is_loss_sum = is_loss_sum

        self.best_acc = 0

        (self.train_set, self.train_loader, self.test_train_set, self.test_train_loader,
         self.test_test_set, self.test_test_loader, self.class_name) = STL10Instance.data(
            self.data_root, batch_size=self.batch_size, input_size=self.input_size)

        self.train_num = self.train_set.__len__()

        self.net = HCResNet(HCBasicBlock, [2, 2, 2, 2], self.low_dim, self.low_dim2,
                            self.low_dim3, self.low_dim4, linear_bias=linear_bias,
                            input_size=self.input_size, conv1_stride=self.conv1_stride).cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

        self._load_model(self.net)

        self.produce_class = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim, ratio=self.ratio1)
        self.produce_class2 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim2, ratio=self.ratio2)
        self.produce_class3 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim3, ratio=self.ratio3)
        self.produce_class4 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim4, ratio=self.ratio4)
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
                       self.test_train_loader, self.test_test_loader, 200, t, loader_n=loader_n)
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
                self.produce_class4.reset()
                for batch_idx, (inputs, _, indexes) in enumerate(self.train_loader):
                    inputs, indexes = inputs.cuda(), indexes.cuda()
                    feature_dict = self.net(inputs)
                    self.produce_class.cal_label(feature_dict[FeatureName.L2norm1], indexes)
                    self.produce_class2.cal_label(feature_dict[FeatureName.L2norm2], indexes)
                    self.produce_class3.cal_label(feature_dict[FeatureName.L2norm3], indexes)
                    self.produce_class4.cal_label(feature_dict[FeatureName.L2norm4], indexes)
                    pass
                Tools.print("Epoch: [{}] 1-{}/{} 2-{}/{} 3-{}/{} 4-{}/{}".format(
                    epoch, self.produce_class.count, self.produce_class.count_2,
                    self.produce_class2.count, self.produce_class2.count_2,
                    self.produce_class3.count, self.produce_class3.count_2,
                    self.produce_class4.count, self.produce_class4.count_2))
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
            avg_loss_4, avg_loss_4_1, avg_loss_4_2 = AverageMeter(), AverageMeter(), AverageMeter()

            for batch_idx, (inputs, _, indexes) in enumerate(self.train_loader):
                inputs, indexes = inputs.cuda(), indexes.cuda()
                self.optimizer.zero_grad()

                feature_dict = self.net(inputs)

                targets = self.produce_class.get_label(indexes)
                targets2 = self.produce_class2.get_label(indexes)
                targets3 = self.produce_class3.get_label(indexes)
                targets4 = self.produce_class4.get_label(indexes)

                params = [_ for _ in self.net.module.parameters()]
                loss_1, loss_1_1, loss_1_2 = self.criterion(
                    feature_dict[FeatureName.Logits1], targets, params[-4], _l1_lambda_)
                loss_2, loss_2_1, loss_2_2 = self.criterion(
                    feature_dict[FeatureName.Logits2], targets2, params[-3], _l1_lambda_)
                loss_3, loss_3_1, loss_3_2 = self.criterion(
                    feature_dict[FeatureName.Logits3], targets3, params[-2], _l1_lambda_)
                loss_4, loss_4_1, loss_4_2 = self.criterion(
                    feature_dict[FeatureName.Logits4], targets4, params[-1], _l1_lambda_)

                avg_loss_1.update(loss_1.item(), inputs.size(0))
                avg_loss_1_1.update(loss_1_1.item(), inputs.size(0))
                avg_loss_1_2.update(loss_1_2.item(), inputs.size(0))
                avg_loss_2.update(loss_2.item(), inputs.size(0))
                avg_loss_2_1.update(loss_2_1.item(), inputs.size(0))
                avg_loss_2_2.update(loss_2_2.item(), inputs.size(0))
                avg_loss_3.update(loss_3.item(), inputs.size(0))
                avg_loss_3_1.update(loss_3_1.item(), inputs.size(0))
                avg_loss_3_2.update(loss_3_2.item(), inputs.size(0))
                avg_loss_4.update(loss_4.item(), inputs.size(0))
                avg_loss_4_1.update(loss_4_1.item(), inputs.size(0))
                avg_loss_4_2.update(loss_4_2.item(), inputs.size(0))

                loss = loss_1 + loss_2 + loss_3 + loss_4 if self.has_l1 \
                    else loss_1_1 + loss_2_1 + loss_3_1 + loss_4_1
                loss.backward()

                self.optimizer.step()
                pass

            Tools.print(
                'Epoch: {}/{} Loss 1: {:.4f}({:.4f}/{:.4f}) Loss 2: {:.4f}({:.4f}/{:.4f}) '
                'Loss 3: {:.4f}({:.4f}/{:.4f}) Loss 4: {:.4f}({:.4f}/{:.4f})'.format(
                    epoch, len(self.train_loader), avg_loss_1.avg, avg_loss_1_1.avg, avg_loss_1_2.avg,avg_loss_2.avg,
                    avg_loss_2_1.avg, avg_loss_2_2.avg, avg_loss_3.avg, avg_loss_3_1.avg, avg_loss_3_2.avg,
                    avg_loss_4.avg, avg_loss_4_1.avg, avg_loss_4_2.avg))
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

    """
    # stl_11_class_1024_4level_512_256_128_no_1600_32_1_l1_sum_0_4321
    81.14(1024, 38718/6873) 81.41(512, 33739/5117) 81.06(256, 30782/4173) 81.11(128, 32504/5089)
    
    # stl_10_class_1024_4level_512_256_128_no_1600_32_1_l1_sum_0_4321_96_1  layer1: strip=1, 1261, 计算量大，时间长
    83.12(1024,29792/6191) 83.16(512,24510/4618) 83.24(256,21888/3774) 83.30(128,18562/2573) K=200
    2019-10-27 13:25:29 Test 1 0 Top1=83.12 Top5=99.46
    2019-10-27 13:25:29 Test 2 0 Top1=83.16 Top5=99.49
    2019-10-27 13:25:29 Test 3 0 Top1=83.24 Top5=99.44
    2019-10-27 13:25:29 Test 4 0 Top1=83.30 Top5=99.42
    2019-10-27 13:25:32 Test 1 0 Top1=86.90 Top5=100.00
    2019-10-27 13:25:32 Test 2 0 Top1=87.70 Top5=100.00
    2019-10-27 13:25:32 Test 3 0 Top1=87.34 Top5=100.00
    2019-10-27 13:25:32 Test 4 0 Top1=87.64 Top5=100.00
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    _start_epoch = 0
    _resume = True

    _max_epoch = 1600
    _learning_rate = 0.01
    _first_epoch, _t_epoch = 200, 100
    _input_size = 96
    _conv1_stride = 1  # 1
    _low_dim, _low_dim2, _low_dim3, _low_dim4 = 1024, 512, 256, 128
    _ratio1, _ratio2, _ratio3, _ratio4 = 4, 3, 2, 1
    _l1_lambda = 0.0
    _is_adjust_lambda = False

    _batch_size = 32
    _is_loss_sum = True
    _has_l1 = True
    _linear_bias = False
    _pre_train = None
    # _pre_train = "./checkpoint/stl_10_class_1024_4level_512_256_128_1600_no_32_1_l1_sum_0/ckpt.t7"
    _name = "stl_10_class_{}_4level_{}_{}_{}_no_{}_{}_{}_l1_sum_{}_{}{}{}{}_{}_{}".format(
        _low_dim, _low_dim2, _low_dim3, _low_dim4, _max_epoch, _batch_size, 0 if _linear_bias else 1,
        1 if _is_adjust_lambda else 0, _ratio1, _ratio2, _ratio3, _ratio4, _input_size, _conv1_stride)
    _checkpoint_path = "./checkpoint/{}/ckpt.t7".format(_name)

    Tools.print()
    Tools.print("name={} input_size={}".format(_name, _input_size))
    Tools.print("low_dim={} low_dim2={} low_dim3={} low_dim4={}".format(_low_dim, _low_dim2, _low_dim3, _low_dim4))
    Tools.print("ratio1={} ratio2={} ratio3={} ratio4={}".format(_ratio1, _ratio2, _ratio3, _ratio4))
    Tools.print("learning_rate={} batch_size={}".format(_learning_rate, _batch_size))
    Tools.print("has_l1={} l1_lambda={} is_adjust_lambda={}".format(_has_l1, _l1_lambda, _is_adjust_lambda))
    Tools.print("pre_train={} checkpoint_path={}".format(_pre_train, _checkpoint_path))
    Tools.print()

    runner = HCRunner(low_dim=_low_dim, low_dim2=_low_dim2, low_dim3=_low_dim3, low_dim4=_low_dim4,
                      ratio1=_ratio1, ratio2=_ratio2, ratio3=_ratio3, ratio4=_ratio4,
                      linear_bias=_linear_bias, has_l1=_has_l1, input_size=_input_size, conv1_stride=_conv1_stride,
                      l1_lambda=_l1_lambda, is_adjust_lambda=_is_adjust_lambda,
                      batch_size=_batch_size, learning_rate=_learning_rate,
                      max_epoch=_max_epoch, t_epoch=_t_epoch, first_epoch=_first_epoch,
                      resume=_resume, pre_train=_pre_train, checkpoint_path=_checkpoint_path)
    Tools.print()
    # acc = runner.test()
    # Tools.print('Random accuracy: {:.2f}'.format(acc * 100))
    # runner.train(start_epoch=_start_epoch, update_epoch=1)
    # Tools.print()
    acc = runner.test(loader_n=2)
    Tools.print('final accuracy: {:.2f}'.format(acc * 100))
    pass
