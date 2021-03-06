import os
import math
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
from cifar_10_tool import AverageMeter, HCLoss, KNN
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
from cifar_10_tool import ProduceClass, FeatureName, Normalize, ImageNetInstance


class MyResNet(ResNet):

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3]):
        super().__init__(block, layers)
        self.out_dim = 512 * block.expansion
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        convB1 = self.maxpool(x)

        convB2 = self.layer1(convB1)
        convB3 = self.layer2(convB2)
        convB4 = self.layer3(convB3)
        convB5 = self.layer4(convB4)

        return convB1, convB2, convB3, convB4, convB5

    pass


class HCResNet(nn.Module):

    def __init__(self, low_dim=8192, low_dim2=4096, low_dim3=2048, low_dim4=1024, linear_bias=True, is_vis=False):
        super().__init__()
        self.is_vis = is_vis

        # self.backbone = MyResNet(block=Bottleneck, layers=[3, 4, 6, 3])
        self.backbone = MyResNet(block=BasicBlock, layers=[2, 2, 2, 2])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_1 = nn.Linear(self.backbone.out_dim, low_dim, bias=linear_bias)
        self.linear_2 = nn.Linear(low_dim, low_dim2, bias=linear_bias)
        self.linear_3 = nn.Linear(low_dim2, low_dim3, bias=linear_bias)
        self.linear_4 = nn.Linear(low_dim3, low_dim4, bias=linear_bias)
        self.l2norm = Normalize(2)
        pass

    def forward(self, x):
        convB1, convB2, convB3, convB4, convB5 = self.backbone(x)

        avgPool = self.avg_pool(convB5)
        avgPool = avgPool.view(avgPool.size(0), -1)
        out_l2norm0 = self.l2norm(avgPool)

        out_logits = self.linear_1(avgPool)
        out_l2norm = self.l2norm(out_logits)

        out_logits2 = self.linear_2(out_logits)
        out_l2norm2 = self.l2norm(out_logits2)

        out_logits3 = self.linear_3(out_logits2)
        out_l2norm3 = self.l2norm(out_logits3)

        out_logits4 = self.linear_4(out_logits3)
        out_l2norm4 = self.l2norm(out_logits4)

        feature_dict = {}

        if self.is_vis:
            feature_dict[FeatureName.x] = x
            feature_dict[FeatureName.ConvB1] = convB1
            feature_dict[FeatureName.ConvB2] = convB2
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

    def __init__(self, low_dim=512, low_dim2=128, low_dim3=10, low_dim4=10,  ratio1=4, ratio2=3, ratio3=2, ratio4=1,
                 batch_size=128, is_loss_sum=False, is_adjust_lambda=False, l1_lambda=0.1, learning_rate=0.01,
                 linear_bias=True, has_l1=False, max_epoch=1000, t_epoch=300, first_epoch=200, resume=False,
                 checkpoint_path="./ckpt.t7", pre_train=None, data_root='./data', worker=16,
                 train_split="train", test_split="val", sample_num=200, temp_size=16):
        self.learning_rate = learning_rate
        self.checkpoint_path = Tools.new_dir(checkpoint_path)
        self.resume = resume
        self.pre_train = pre_train
        self.data_root = data_root
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.temp_size = temp_size

        self.low_dim = low_dim
        self.low_dim2 = low_dim2
        self.low_dim3 = low_dim3
        self.low_dim4 = low_dim4
        self.ratio1 = ratio1
        self.ratio2 = ratio2
        self.ratio3 = ratio3
        self.ratio4 = ratio4

        self.t_epoch = t_epoch
        self.max_epoch = max_epoch
        self.first_epoch = first_epoch
        self.linear_bias = linear_bias
        self.has_l1 = has_l1
        self.l1_lambda = l1_lambda
        self.is_adjust_lambda = is_adjust_lambda
        self.is_loss_sum = is_loss_sum
        self.worker = worker

        self.best_acc = 0

        _test_dir = os.path.join(self.data_root, test_split)
        _train_dir = os.path.join(self.data_root, train_split)
        (self.train_set, self.train_loader, self.test_set, self.test_loader, self.train_set_for_test,
         self.train_loader_for_test) = ImageNetInstance.data(train_root=_train_dir, test_root=_test_dir,
                                                             batch_size=self.batch_size, worker=self.worker,
                                                             output_size=224, sample_num=self.sample_num)
        self.train_num = self.train_set.__len__()

        self.net = HCResNet(self.low_dim, self.low_dim2, self.low_dim3, self.low_dim4, linear_bias=linear_bias)
        self.low_dim_list = [self.net.backbone.out_dim, self.low_dim, self.low_dim2, self.low_dim3, self.low_dim4]

        #############################################################
        self.net = nn.DataParallel(self.net)
        self.net = self.net.cuda()
        self._load_model(self.net)

        cudnn.benchmark = True
        #############################################################

        self.produce_class11 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim, ratio=self.ratio1)
        self.produce_class21 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim2, ratio=self.ratio2)
        self.produce_class31 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim3, ratio=self.ratio3)
        self.produce_class41 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim4, ratio=self.ratio4)
        self.produce_class12 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim, ratio=self.ratio1)
        self.produce_class22 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim2, ratio=self.ratio2)
        self.produce_class32 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim3, ratio=self.ratio3)
        self.produce_class42 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim4, ratio=self.ratio4)
        self.produce_class11.init()
        self.produce_class21.init()
        self.produce_class31.init()
        self.produce_class41.init()
        self.produce_class12.init()
        self.produce_class22.init()
        self.produce_class32.init()
        self.produce_class42.init()

        self.criterion = HCLoss().cuda()  # define loss function
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        pass

    def _adjust_learning_rate2(self, epoch):

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

    def _adjust_learning_rate(self, epoch):

        def _get_lr(_base_lr, now_epoch, _t_epoch=self.t_epoch, _eta_min=1e-05):
            return _eta_min + (_base_lr - _eta_min) * (1 + math.cos(math.pi * now_epoch / _t_epoch)) / 2

        t_epoch = self.t_epoch
        first_epoch = self.first_epoch
        if epoch < first_epoch + self.t_epoch * 0:  # 0-100
            learning_rate = self.learning_rate
        elif epoch < first_epoch + t_epoch * 1:  # 100-200
            learning_rate = self.learning_rate / 3
        elif epoch < first_epoch + t_epoch * 2:  # 200-300
            learning_rate = self.learning_rate / 10
        elif epoch < first_epoch + t_epoch * 3:  # 300-400
            learning_rate = _get_lr(self.learning_rate / 3.0, epoch - first_epoch - t_epoch * 2)
        else:  # 400-500
            learning_rate = _get_lr(self.learning_rate / 10.0, epoch - first_epoch - t_epoch * 3)
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

    def test(self, epoch=0, t=0.1, loader_n=1, k=50):
        all_top1_list, all_top5_list = KNN.knn(
            epoch, self.net, [self.low_dim_list[0]], self.train_loader_for_test,
            self.test_loader, k, t, loader_n=loader_n, temp_size=self.temp_size,
            return_all_acc=True, temp_num_workers=self.worker)
        return all_top1_list, all_top5_list

    def _train_one_epoch(self, epoch, test_freq=100):
        # Train
        try:
            self.net.train()
            _learning_rate_ = self._adjust_learning_rate(epoch)
            _l1_lambda_ = self._adjust_l1_lambda(epoch)
            Tools.print('Epoch: [{}] lr={} lambda={}'.format(epoch, _learning_rate_, _l1_lambda_))

            avg_loss_1, avg_loss_1_1, avg_loss_1_2 = AverageMeter(), AverageMeter(), AverageMeter()
            avg_loss_2, avg_loss_2_1, avg_loss_2_2 = AverageMeter(), AverageMeter(), AverageMeter()
            avg_loss_3, avg_loss_3_1, avg_loss_3_2 = AverageMeter(), AverageMeter(), AverageMeter()
            avg_loss_4, avg_loss_4_1, avg_loss_4_2 = AverageMeter(), AverageMeter(), AverageMeter()

            self.produce_class11.reset()
            self.produce_class21.reset()
            self.produce_class31.reset()
            self.produce_class41.reset()
            for batch_idx, (inputs, _, indexes) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                inputs, indexes = inputs.cuda(), indexes.cuda()
                self.optimizer.zero_grad()

                feature_dict = self.net(inputs)

                self.produce_class11.cal_label(feature_dict[FeatureName.L2norm1], indexes)
                self.produce_class21.cal_label(feature_dict[FeatureName.L2norm2], indexes)
                self.produce_class31.cal_label(feature_dict[FeatureName.L2norm3], indexes)
                self.produce_class41.cal_label(feature_dict[FeatureName.L2norm4], indexes)

                targets = self.produce_class12.get_label(indexes)
                targets2 = self.produce_class22.get_label(indexes)
                targets3 = self.produce_class32.get_label(indexes)
                targets4 = self.produce_class42.get_label(indexes)

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

                loss = loss_1 + loss_2 + loss_3 + loss_4 if self.has_l1 else loss_1_1 + loss_2_1 + loss_3_1 + loss_4_1
                loss.backward()

                self.optimizer.step()
                pass

            classes = self.produce_class12.classes
            self.produce_class12.classes = self.produce_class11.classes
            self.produce_class11.classes = classes
            classes = self.produce_class22.classes
            self.produce_class22.classes = self.produce_class21.classes
            self.produce_class21.classes = classes
            classes = self.produce_class32.classes
            self.produce_class32.classes = self.produce_class31.classes
            self.produce_class31.classes = classes
            classes = self.produce_class42.classes
            self.produce_class42.classes = self.produce_class41.classes
            self.produce_class41.classes = classes
            Tools.print("Train: [{}] 1-{}/{}".format(epoch, self.produce_class11.count, self.produce_class11.count_2))
            Tools.print("Train: [{}] 2-{}/{}".format(epoch, self.produce_class21.count, self.produce_class21.count_2))
            Tools.print("Train: [{}] 3-{}/{}".format(epoch, self.produce_class31.count, self.produce_class31.count_2))
            Tools.print("Train: [{}] 4-{}/{}".format(epoch, self.produce_class41.count, self.produce_class41.count_2))

            Tools.print(
                'Train: [{}] {} Loss 1: {:.4f}({:.4f}/{:.4f}) Loss 2: {:.4f}({:.4f}/{:.4f}) '
                'Loss 3: {:.4f}({:.4f}/{:.4f}) Loss 4: {:.4f}({:.4f}/{:.4f})'.format(
                    epoch, len(self.train_loader),
                    avg_loss_1.avg, avg_loss_1_1.avg, avg_loss_1_2.avg,
                    avg_loss_2.avg, avg_loss_2_1.avg, avg_loss_2_2.avg,
                    avg_loss_3.avg, avg_loss_3_1.avg, avg_loss_3_2.avg,
                    avg_loss_4.avg, avg_loss_4_1.avg, avg_loss_4_2.avg))
        finally:
            pass

        Tools.print('Saving..')
        state = {'net': self.net.state_dict(), 'acc': 0, 'epoch': epoch}
        new_path = os.path.split(self.checkpoint_path)
        torch.save(state, os.path.join(new_path[0], "{}_{}".format(epoch, new_path[1])))

        # Test
        if epoch % test_freq == 0:
            Tools.print("Test:  [{}] .......".format(epoch))
            all_top1_list, all_top5_list = self.test(epoch=epoch)
            Tools.print('Epoch: [{}] {}'.format(epoch, (all_top1_list, all_top5_list)))

            if all_top5_list[0][0] > self.best_acc:
                Tools.print('Saving..')
                state = {'net': self.net.state_dict(), 'acc': all_top5_list[0][0], 'epoch': epoch}
                torch.save(state, self.checkpoint_path)
                self.best_acc = all_top5_list[0][0]
                pass

            Tools.print('Epoch: [{}] best accuracy: {:.2f}'.format(epoch, self.best_acc))
            pass

        pass

    def train(self, start_epoch):

        if start_epoch > 0:
            self.net.eval()
            Tools.print("Update label {} .......".format(start_epoch))
            self.produce_class11.reset()
            self.produce_class21.reset()
            self.produce_class31.reset()
            self.produce_class41.reset()
            with torch.no_grad():
                for batch_idx, (inputs, _, indexes) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                    inputs, indexes = inputs.cuda(), indexes.cuda()
                    feature_dict = self.net(inputs)
                    self.produce_class11.cal_label(feature_dict[FeatureName.L2norm1], indexes)
                    self.produce_class21.cal_label(feature_dict[FeatureName.L2norm2], indexes)
                    self.produce_class31.cal_label(feature_dict[FeatureName.L2norm3], indexes)
                    self.produce_class41.cal_label(feature_dict[FeatureName.L2norm4], indexes)
                    pass
            classes = self.produce_class12.classes
            self.produce_class12.classes = self.produce_class11.classes
            self.produce_class11.classes = classes
            classes = self.produce_class22.classes
            self.produce_class22.classes = self.produce_class21.classes
            self.produce_class21.classes = classes
            classes = self.produce_class32.classes
            self.produce_class32.classes = self.produce_class31.classes
            self.produce_class31.classes = classes
            classes = self.produce_class42.classes
            self.produce_class42.classes = self.produce_class41.classes
            self.produce_class41.classes = classes
            Tools.print("Train: [{}] 1-{}/{}".format(start_epoch, self.produce_class11.count, self.produce_class11.count_2))
            Tools.print("Train: [{}] 2-{}/{}".format(start_epoch, self.produce_class21.count, self.produce_class21.count_2))
            Tools.print("Train: [{}] 3-{}/{}".format(start_epoch, self.produce_class31.count, self.produce_class31.count_2))
            Tools.print("Train: [{}] 4-{}/{}".format(start_epoch, self.produce_class41.count, self.produce_class41.count_2))
            pass

        for epoch in range(start_epoch, self.max_epoch):
            Tools.print()
            self._train_one_epoch(epoch)
            pass

        pass

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

    """
    2020-05-30 01:17:31 Epoch: [68] lr=0.0002477738420031906 lambda=0.0
    2020-05-30 02:35:00 Train: [68] 1-739736/40567
    2020-05-30 02:35:00 Train: [68] 2-761612/34539
    2020-05-30 02:35:00 Train: [68] 3-773628/36000
    2020-05-30 02:35:00 Train: [68] 4-784377/43882
    2020-05-30 02:35:00 Train: [68] 20019 Loss 1: 2.1412(2.1412/3.9069) Loss 2: 2.1771(2.1771/3.2457) Loss 3: 2.1997(2.1997/4.3125) Loss 4: 2.2124(2.2124/7.7153)
    2020-05-30 02:35:00 Test:  [68] .......
    2020-05-30 02:51:44 Test:  [68] 0 Top1=0.10 Top5=29.76
    2020-05-30 02:51:44 Epoch: [68] ([[0.098]], [[29.762]])
    2020-05-30 02:51:44 Saving..
    2020-05-30 02:51:44 Epoch: [68] best accuracy: 29.76
    
    2020-06-21 10:47:35 Train: [499] 1-499136/37922
    2020-06-21 10:47:35 Train: [499] 2-490279/35241
    2020-06-21 10:47:35 Train: [499] 3-488621/34436
    2020-06-21 10:47:35 Train: [499] 4-494024/38869
    2020-06-21 10:47:35 Train: [499] 2503 Loss 1: 1.3763(1.3763/3.8635) Loss 2: 1.3242(1.3242/3.4231) Loss 3: 1.3251(1.3251/4.2186) Loss 4: 1.3382(1.3382/7.1630)
    2020-06-21 10:49:11 Test:  [0] 0 Top1=0.09 Top5=48.69
    """

    _start_epoch = 1
    _max_epoch = 500
    _learning_rate = 0.01
    _first_epoch, _t_epoch = 100, 50
    _low_dim, _low_dim2, _low_dim3, _low_dim4 = 1024 * 4, 1024 * 3, 1024 * 2, 1024 * 1
    _ratio1, _ratio2, _ratio3, _ratio4 = 4, 3, 2, 1
    _l1_lambda = 0.0
    _is_adjust_lambda = False
    _test_sample_num = 100

    _batch_size = 256 * 2

    # _data_root_path = '/mnt/4T/Data/ILSVRC17/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC'
    # _data_root_path = '/mnt/4T/Data/data/ImageNet/ILSVRC2015/Data/CLS-LOC'
    _data_root_path = '/media/ubuntu/ALISURE-SSD/data/ImageNet/ILSVRC2015/Data/CLS-LOC'
    _test_split = "val"
    _test_temp_size = 128 * 1
    _is_loss_sum = True
    _has_l1 = True
    _worker = 40
    _linear_bias = False
    _resume = False
    _pre_train = "./checkpoint/imagenet/res18_class_4level_4096_3072_2048_1024_no_160_64_1_l1_sum_0_4321/74_0.088_28.838_ckpt.t7"
    _name = "res18_class_4level_{}_{}_{}_{}_no_{}_{}_{}_l1_sum_{}_{}{}{}{}".format(
        _low_dim, _low_dim2, _low_dim3, _low_dim4, _max_epoch, _batch_size, 0 if _linear_bias else 1,
        1 if _is_adjust_lambda else 0, _ratio1, _ratio2, _ratio3, _ratio4)
    _checkpoint_path = "./checkpoint/imagenet/{}/ckpt.t7".format(_name)

    Tools.print()
    Tools.print("name={}".format(_name))
    Tools.print("low_dim={} low_dim2={} low_dim3={} low_dim4={}".format(_low_dim, _low_dim2, _low_dim3, _low_dim4))
    Tools.print("ratio1={} ratio2={} ratio3={} ratio4={}".format(_ratio1, _ratio2, _ratio3, _ratio4))
    Tools.print("learning_rate={} batch_size={}".format(_learning_rate, _batch_size))
    Tools.print("has_l1={} l1_lambda={} is_adjust_lambda={}".format(_has_l1, _l1_lambda, _is_adjust_lambda))
    Tools.print("pre_train={} checkpoint_path={}".format(_pre_train, _checkpoint_path))
    Tools.print()

    runner = HCRunner(low_dim=_low_dim, low_dim2=_low_dim2, low_dim3=_low_dim3, low_dim4=_low_dim4, ratio1=_ratio1,
                      ratio2=_ratio2, ratio3=_ratio3, ratio4=_ratio4, linear_bias=_linear_bias, has_l1=_has_l1,
                      l1_lambda=_l1_lambda, is_adjust_lambda=_is_adjust_lambda, is_loss_sum=_is_loss_sum,
                      batch_size=_batch_size, learning_rate=_learning_rate, max_epoch=_max_epoch, t_epoch=_t_epoch,
                      first_epoch=_first_epoch, resume=_resume, pre_train=_pre_train, sample_num=_test_sample_num,
                      checkpoint_path=_checkpoint_path, data_root=_data_root_path, test_split=_test_split,
                      temp_size=_test_temp_size, worker=_worker)
    Tools.print()
    # acc = runner.test()
    # Tools.print('Random accuracy: {}'.format(acc))
    runner.train(start_epoch=_start_epoch)
    Tools.print()
    acc = runner.test(loader_n=2)
    Tools.print('Final accuracy: {}'.format(acc))
    pass
