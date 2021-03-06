import os
import math
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from alisuretool.Tools import Tools
from cifar_10_tool import HCBasicBlock, AverageMeter, HCLoss, KNN
from torchvision.models.resnet import conv1x1, conv3x3, BasicBlock, Bottleneck
from cifar_10_tool import ProduceClass, FeatureName, Normalize, CIFAR10Instance


class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 16
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 8
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 4
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        pass

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x

    pass


class HCAlexNet(nn.Module):

    def __init__(self, low_dim=512, low_dim2=128, low_dim3=10, low_dim4=10, low_dim5=10, linear_bias=True):
        super(HCAlexNet, self).__init__()
        self.backbone = AlexNet()

        self.linear_1024 = nn.Linear(256, low_dim, bias=linear_bias)
        self.linear_512 = nn.Linear(low_dim, low_dim2, bias=linear_bias)
        self.linear_256 = nn.Linear(low_dim2, low_dim3, bias=linear_bias)
        self.linear_128 = nn.Linear(low_dim3, low_dim4, bias=linear_bias)
        self.linear_64 = nn.Linear(low_dim4, low_dim5, bias=linear_bias)
        self.l2norm = Normalize(2)
        pass

    def forward(self, x):
        out = self.backbone(x)

        out = out.view(out.size(0), -1)
        out_l2norm0 = self.l2norm(out)

        out_logits = self.linear_1024(out)
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
        feature_dict[FeatureName.Logits0] = out
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
        self.low_dim_list = [256, self.low_dim, self.low_dim2, self.low_dim3, self.low_dim4, self.low_dim5]
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
        self.train_num = self.train_set.__len__()

        self.net = HCAlexNet(self.low_dim, self.low_dim2, self.low_dim3,
                             self.low_dim4, self.low_dim5, linear_bias=linear_bias).cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

        self._load_model(self.net)

        self.produce_class = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim, ratio=self.ratio1)
        self.produce_class2 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim2, ratio=self.ratio2)
        self.produce_class3 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim3, ratio=self.ratio3)
        self.produce_class4 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim4, ratio=self.ratio4)
        self.produce_class5 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim5, ratio=self.ratio5)
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
                self.produce_class4.reset()
                self.produce_class5.reset()
                for batch_idx, (inputs, _, indexes) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                    inputs, indexes = inputs.cuda(), indexes.cuda()
                    feature_dict = self.net(inputs)
                    self.produce_class.cal_label(feature_dict[FeatureName.L2norm1], indexes)
                    self.produce_class2.cal_label(feature_dict[FeatureName.L2norm2], indexes)
                    self.produce_class3.cal_label(feature_dict[FeatureName.L2norm3], indexes)
                    self.produce_class4.cal_label(feature_dict[FeatureName.L2norm4], indexes)
                    self.produce_class5.cal_label(feature_dict[FeatureName.L2norm5], indexes)
                    pass
                Tools.print("Epoch: [{}] 1-{}/{} 2-{}/{} 3-{}/{} 4-{}/{} 5-{}/{}".format(
                    epoch, self.produce_class.count, self.produce_class.count_2,
                    self.produce_class2.count, self.produce_class2.count_2,
                    self.produce_class3.count, self.produce_class3.count_2,
                    self.produce_class4.count, self.produce_class4.count_2,
                    self.produce_class5.count, self.produce_class5.count_2))
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
            avg_loss_5, avg_loss_5_1, avg_loss_5_2 = AverageMeter(), AverageMeter(), AverageMeter()

            for batch_idx, (inputs, _, indexes) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                inputs, indexes = inputs.cuda(), indexes.cuda()
                self.optimizer.zero_grad()

                feature_dict = self.net(inputs)

                targets = self.produce_class.get_label(indexes)
                targets2 = self.produce_class2.get_label(indexes)
                targets3 = self.produce_class3.get_label(indexes)
                targets4 = self.produce_class4.get_label(indexes)
                targets5 = self.produce_class5.get_label(indexes)

                params = [_ for _ in self.net.module.parameters()]
                loss_1, loss_1_1, loss_1_2 = self.criterion(
                    feature_dict[FeatureName.Logits1], targets, params[-5], _l1_lambda_)
                loss_2, loss_2_1, loss_2_2 = self.criterion(
                    feature_dict[FeatureName.Logits2], targets2, params[-4], _l1_lambda_)
                loss_3, loss_3_1, loss_3_2 = self.criterion(
                    feature_dict[FeatureName.Logits3], targets3, params[-3], _l1_lambda_)
                loss_4, loss_4_1, loss_4_2 = self.criterion(
                    feature_dict[FeatureName.Logits4], targets4, params[-2], _l1_lambda_)
                loss_5, loss_5_1, loss_5_2 = self.criterion(
                    feature_dict[FeatureName.Logits5], targets5, params[-1], _l1_lambda_)

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
                avg_loss_5.update(loss_5.item(), inputs.size(0))
                avg_loss_5_1.update(loss_5_1.item(), inputs.size(0))
                avg_loss_5_2.update(loss_5_2.item(), inputs.size(0))

                loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 if self.has_l1 \
                    else loss_1_1 + loss_2_1 + loss_3_1 + loss_4_1 + loss_5_1
                loss.backward()

                self.optimizer.step()
                pass

            Tools.print(
                'Epoch: {}/{} Loss 1: {:.4f}({:.4f}/{:.4f}) Loss 2: {:.4f}({:.4f}/{:.4f}) '
                'Loss 3: {:.4f}({:.4f}/{:.4f}) Loss 4: {:.4f}({:.4f}/{:.4f}) Loss 5: {:.4f}({:.4f}/{:.4f})'.format(
                    epoch, len(self.train_loader),
                    avg_loss_1.avg, avg_loss_1_1.avg, avg_loss_1_2.avg,
                    avg_loss_2.avg, avg_loss_2_1.avg, avg_loss_2_2.avg,
                    avg_loss_3.avg, avg_loss_3_1.avg, avg_loss_3_2.avg,
                    avg_loss_4.avg, avg_loss_4_1.avg, avg_loss_4_2.avg,
                    avg_loss_5.avg, avg_loss_5_1.avg, avg_loss_5_2.avg))
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
            Tools.print('Epoch: [{}] best accuracy: {:.2f}'.format(epoch, self.best_acc))
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
    # 11_class_1024_5level_512_256_128_64_no_1600_32_1_l1_sum_0_54321
    88.38(1024) 88.33(512) 88.17(256) 88.14(128) 88.50(64)
    
    2020-05-28 10:42:12 Update label 1468 .......
    2020-05-28 10:42:47 Epoch: [1468] 1-29428/2696 2-28533/3114 3-26847/3319 4-25778/3309 5-26085/3522
    2020-05-28 10:42:47 Epoch: 1468 lr=4.3942665617160856e-05 lambda=0.0
    2020-05-28 10:43:32 Epoch: 1468/1563 Loss 1: 2.2644(2.2644/2.5462) Loss 2: 2.0609(2.0609/2.4532) Loss 3: 1.8812(1.8812/3.6765) Loss 4: 1.7555(1.7555/3.6566) Loss 5: 1.7820(1.7820/4.2388)
    2020-05-28 10:43:32 Test [1468] .......
    2020-05-28 10:44:01 Test:  [1468] 0 Top1=72.55 Top5=97.77
    2020-05-28 10:44:01 Test:  [1468] 1 Top1=70.60 Top5=97.39
    2020-05-28 10:44:01 Test:  [1468] 2 Top1=71.12 Top5=97.49
    2020-05-28 10:44:01 Test:  [1468] 3 Top1=71.09 Top5=97.45
    2020-05-28 10:44:01 Test:  [1468] 4 Top1=70.60 Top5=97.34
    2020-05-28 10:44:01 Test:  [1468] 5 Top1=69.90 Top5=97.28
    2020-05-28 10:44:01 Saving..
    2020-05-28 10:44:01 Epoch: [1468] best accuracy: 69.90
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
    _resume = False
    _pre_train = None
    # _pre_train = "./checkpoint/cifar10/alexnet_class_1024_5level_512_256_128_64_no_1600_32_1_l1_sum_0_54321/ckpt.t7"
    _name = "alexnet_class_{}_5level_{}_{}_{}_{}_no_{}_{}_{}_l1_sum_{}_{}{}{}{}{}".format(
        _low_dim, _low_dim2, _low_dim3, _low_dim4, _low_dim5, _max_epoch, _batch_size,
        0 if _linear_bias else 1, 1 if _is_adjust_lambda else 0, _ratio1, _ratio2, _ratio3, _ratio4, _ratio5)
    _checkpoint_path = "./checkpoint/cifar10/{}/ckpt.t7".format(_name)

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
                      linear_bias=_linear_bias, has_l1=_has_l1, l1_lambda=_l1_lambda,
                      is_adjust_lambda=_is_adjust_lambda, is_loss_sum=_is_loss_sum, batch_size=_batch_size,
                      learning_rate=_learning_rate, max_epoch=_max_epoch, t_epoch=_t_epoch, first_epoch=_first_epoch,
                      resume=_resume, pre_train=_pre_train, checkpoint_path=_checkpoint_path,
                      data_root="/media/test/ALISURE/data/cifar10")
    Tools.print()
    acc = runner.test()
    Tools.print('Random accuracy: {:.2f}'.format(acc))
    runner.train(start_epoch=_start_epoch, update_epoch=1)
    Tools.print()
    acc = runner.test(loader_n=2)
    Tools.print('final accuracy: {:.2f}'.format(acc))
    pass
