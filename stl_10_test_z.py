import os
import torch
import torch.nn as nn
import torch.optim as optimizer
from alisuretool.Tools import Tools
from cifar_10_tool import HCBasicBlock, AverageMeter, STL10Instance, FeatureName


class MultipleNonLinearClassifier(nn.Module):

    def __init__(self, input_size_list, output_size=10):
        super(MultipleNonLinearClassifier, self).__init__()

        self.linear_1 = nn.Linear(input_size_list[0], input_size_list[1], bias=False)
        self.bn_1 = nn.BatchNorm1d(input_size_list[1])
        self.relu_1 = nn.ReLU(inplace=True)

        self.linear_2 = nn.Linear(input_size_list[1], input_size_list[2], bias=False)
        self.bn_2 = nn.BatchNorm1d(input_size_list[2])
        self.relu_2 = nn.ReLU(inplace=True)

        self.linear_3 = nn.Linear(input_size_list[2], output_size)
        pass

    def forward(self, inputs):
        out = self.relu_1(self.bn_1(self.linear_1(inputs)))
        out = self.relu_2(self.bn_2(self.linear_2(out)))
        out = self.linear_3(out)
        return out

    pass


class Classifier(nn.Module):

    def __init__(self, low_dim, input_size_or_list, output_size=10, input_size=32, conv1_stride=1,
                 feature_name=FeatureName.L2norm0, is_fine_tune=False, linear_bias=True):
        super(Classifier, self).__init__()

        self.feature_name = feature_name
        self.is_fine_tune = is_fine_tune
        self.input_size = input_size
        self.conv1_stride = conv1_stride

        self.attention = HCResNet(HCBasicBlock, [2, 2, 2, 2], *low_dim, linear_bias=linear_bias,
                                  input_size=self.input_size, conv1_stride=self.conv1_stride, is_vis=True).cuda()

        if not self.is_fine_tune:
            # 3, 15, 30, 45, 60
            if self.feature_name == FeatureName.ConvB2:
                for index, p in enumerate(self.parameters()):
                    if index < 15:
                        p.requires_grad = False
                    pass
                self.feature_name = FeatureName.L2norm0
                pass
            elif self.feature_name == FeatureName.ConvB3:
                for index, p in enumerate(self.parameters()):
                    if index < 30:
                        p.requires_grad = False
                    pass
                self.feature_name = FeatureName.L2norm0
                pass
            elif self.feature_name == FeatureName.ConvB4:
                for index, p in enumerate(self.parameters()):
                    if index < 45:
                        p.requires_grad = False
                    pass
                self.feature_name = FeatureName.L2norm0
                pass
            else:
                for p in self.parameters():
                    p.requires_grad = False
                    pass
                pass
            pass

        assert isinstance(input_size_or_list, list) and len(input_size_or_list) >= 1
        self.linear = MultipleNonLinearClassifier(input_size_or_list, output_size)
        pass

    def forward(self, inputs):
        out = self.attention(inputs)
        out = out[self.feature_name]
        out = self.linear(out)
        return out

    pass


class ClassierRunner(object):

    def __init__(self, net, learning_rate=0.01, max_epoch=1000, resume=False, input_size=32, is_fine_tune=False,
                 pre_train_path=None, checkpoint_path="./classier.t7", data_root='./data'):
        self.learning_rate = learning_rate
        self.checkpoint_path = Tools.new_dir(checkpoint_path)
        self.pre_train_path = pre_train_path
        self.resume = resume
        self.is_fine_tune = is_fine_tune
        self.data_root = data_root
        self.input_size = input_size

        self.best_acc = 0

        if self.is_fine_tune:
            self.learning_rate = 0.001 if pre_train_path else 0.01
            pass

        (self.train_set, self.train_loader, self.test_train_set, self.test_train_loader,
         self.test_test_set, self.test_test_loader, _) = STL10Instance.data(
            self.data_root, input_size=self.input_size, is_test_train_shuffle=True)
        self.train_num = self.train_set.__len__()

        self.net = net.cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

        self._load_model(self.net)

        self.criterion = nn.CrossEntropyLoss().cuda()  # define loss function
        self.optimizer = optimizer.SGD(filter(lambda p: p.requires_grad, self.net.parameters()),
                                       lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)

        self.max_epoch = max_epoch
        pass

    def _adjust_learning_rate(self, epoch):
        if epoch < 50:
            learning_rate = self.learning_rate
        elif epoch < 100:
            learning_rate = self.learning_rate * 0.1
        elif epoch < 150:
            learning_rate = self.learning_rate * 0.01
        else:
            learning_rate = self.learning_rate * 0.001
            pass

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
            pass
        return learning_rate

    def _load_model(self, net):
        # pre train
        if self.pre_train_path:
            Tools.print('==> Pre train from checkpoint {} ..'.format(self.pre_train_path))
            checkpoint = torch.load(self.pre_train_path)
            checkpoint_value = {key.replace("module.", "module.attention."): checkpoint["net"][key]
                                for key in checkpoint["net"].keys()}
            if 'acc' in checkpoint.keys() and 'epoch' in checkpoint.keys():
                best_acc = checkpoint['acc']
                best_epoch = checkpoint['epoch']
                Tools.print("{} {}".format(best_acc, best_epoch))
            net.load_state_dict(checkpoint_value, strict=False)
            pass

        # Load checkpoint.
        if self.resume:
            Tools.print('==> Resuming from checkpoint {} ..'.format(self.checkpoint_path))
            checkpoint = torch.load(self.checkpoint_path)
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            best_epoch = checkpoint['epoch']
            Tools.print("{} {}".format(best_acc, best_epoch))
            pass
        pass

    def test(self, epoch=0, is_test_test=True):
        self.net.eval()
        total = 0
        correct = 0
        loader = self.test_test_loader if is_test_test else self.test_train_loader
        for batch_idx, (inputs, targets, _) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = self.net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pass
        Tools.print("Epoch {} {} {}/({}-{})".format(epoch, " Test" if is_test_test else "Train",
                                                    correct/total, correct, total))
        return correct / total * 100

    def _train_one_epoch(self, epoch, test_per_epoch=3):
        # Train
        Tools.print()
        self.net.train()

        learning_rate = self._adjust_learning_rate(epoch)
        Tools.print("Epoch: [{}] lr={}".format(epoch, learning_rate))

        avg_loss = AverageMeter()
        for batch_idx, (inputs, targets, _) in enumerate(self.test_train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer.zero_grad()

            out_logits = self.net(inputs)

            loss = self.criterion(out_logits, targets)
            avg_loss.update(loss.item(), inputs.size(0))
            loss.backward()
            self.optimizer.step()
            pass

        Tools.print('Epoch: [{}-{}] Loss: {avg_loss.val:.4f} ({avg_loss.avg:.4f})'.format(
            epoch, len(self.test_train_loader), avg_loss=avg_loss))

        # Test
        if epoch % test_per_epoch == 0:
            _ = self.test(epoch, is_test_test=False)
            _test_acc = self.test(epoch, is_test_test=True)

            if _test_acc > self.best_acc:
                Tools.print('Saving..')
                state = {'net': self.net.state_dict(), 'acc': _test_acc, 'epoch': epoch}
                torch.save(state, self.checkpoint_path)
                self.best_acc = _test_acc
                pass
            Tools.print('best accuracy: {:.2f}'.format(self.best_acc))
            pass
        pass

    def train(self, start_epoch, test_per_epoch=3):
        for epoch in range(start_epoch, self.max_epoch):
            self._train_one_epoch(epoch, test_per_epoch=test_per_epoch)
            pass
        pass

    pass


if __name__ == '__main__':

    _image_size = 96
    _conv1_stride = 1

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 1
    # _low_dim = [128]
    # _name = "stl_10_class_128_1level_1600_no_32_1_l1_sum_0_1_96_1"
    # from stl_10_1level_z import HCResNet
    # # _feature_list = [[FeatureName.ConvB3, 512], [FeatureName.ConvB4, 512],
    # #                  [FeatureName.Logits0, 512], [FeatureName.Logits1, _low_dim[0]]]
    # _feature_list = [[FeatureName.ConvB3, 512], [FeatureName.ConvB4, 512],
    #                  [FeatureName.L2norm0, 512], [FeatureName.L2norm1, _low_dim[0]]]
    # 2
    _low_dim = [1024, 128]
    _name = "stl_10_class_1024_2level_128_1600_no_32_1_l1_sum_0_21_96_1"
    from stl_10_2level_z import HCResNet
    # _feature_list = [[FeatureName.ConvB3, 512], [FeatureName.ConvB4, 512],
    #                  [FeatureName.Logits0, 512], [FeatureName.Logits1, _low_dim[0]],
    #                  [FeatureName.Logits2, _low_dim[1]]]
    _feature_list = [[FeatureName.ConvB3, 512], [FeatureName.ConvB4, 512],
                     [FeatureName.L2norm0, 512], [FeatureName.L2norm1, _low_dim[0]],
                     [FeatureName.L2norm2, _low_dim[1]]]
    # 3
    # _low_dim = [1024, 256, 64]
    # _name = "11_class_1024_3level_256_64_1600_no_32_1_l1_sum_0_321"
    # from cifar_10_3level_z import HCResNet
    # # _feature_list = [[FeatureName.ConvB3, 512], [FeatureName.ConvB4, 512],
    # #                  [FeatureName.Logits0, 512], [FeatureName.Logits1, _low_dim[0]],
    # #                  [FeatureName.Logits2, _low_dim[1]], [FeatureName.Logits3, _low_dim[2]]]
    # _feature_list = [[FeatureName.ConvB3, 512], [FeatureName.ConvB4, 512],
    #                  [FeatureName.L2norm0, 512], [FeatureName.L2norm1, _low_dim[0]],
    #                  [FeatureName.L2norm2, _low_dim[1]], [FeatureName.L2norm3, _low_dim[2]]]
    # 4
    # _low_dim = [1024, 512, 256, 128]
    # _name = "11_class_1024_4level_512_256_128_no_1600_32_1_l1_sum_0_4321"
    # from cifar_10_4level_z import HCResNet
    # # _feature_list = [[FeatureName.ConvB3, 512], [FeatureName.ConvB4, 512],
    # #                  [FeatureName.Logits0, 512], [FeatureName.Logits1, _low_dim[0]],
    # #                  [FeatureName.Logits2, _low_dim[1]], [FeatureName.Logits3, _low_dim[2]],
    # #                  [FeatureName.Logits4, _low_dim[3]]]
    # _feature_list = [[FeatureName.ConvB3, 512], [FeatureName.ConvB4, 512],
    #                  [FeatureName.L2norm0, 512], [FeatureName.L2norm1, _low_dim[0]],
    #                  [FeatureName.L2norm2, _low_dim[1]], [FeatureName.L2norm3, _low_dim[2]],
    #                  [FeatureName.L2norm4, _low_dim[3]]]
    # 5
    # _low_dim = [1024, 512, 256, 128, 64]
    # _name = "11_class_1024_5level_512_256_128_64_no_1600_32_1_l1_sum_0_54321"
    # from cifar_10_5level_z import HCResNet
    # _feature_list = [[FeatureName.ConvB3, 512], [FeatureName.ConvB4, 512],
    #                  [FeatureName.Logits0, 512], [FeatureName.Logits1, _low_dim[0]],
    #                  [FeatureName.Logits2, _low_dim[1]], [FeatureName.Logits3, _low_dim[2]],
    #                  [FeatureName.Logits4, _low_dim[3]], [FeatureName.Logits5, _low_dim[4]]]
    # _feature_list = [[FeatureName.ConvB3, 512], [FeatureName.ConvB4, 512],
    #                  [FeatureName.L2norm0, 512], [FeatureName.L2norm1, _low_dim[0]],
    #                  [FeatureName.L2norm2, _low_dim[1]], [FeatureName.L2norm3, _low_dim[2]],
    #                  [FeatureName.L2norm4, _low_dim[3]], [FeatureName.L2norm5, _low_dim[4]]]

    _which = 4
    _feature_name = _feature_list[_which][0]
    _input_size = _feature_list[_which][1]

    _is_fine_tune = False

    _start_epoch = 0  # train epoch
    _max_epoch = 200
    _linear_bias = False

    # _pre_train_path = None
    _pre_train_path = "./checkpoint/{}/ckpt.t7".format(_name)
    _checkpoint_path_classier = "./checkpoint/{}/classier_{}_{}_{}_{}.t7".format(
        _name, _input_size, _feature_name, 1 if _is_fine_tune else 0, 1 if _pre_train_path else 0)

    Tools.print()
    Tools.print("input_size={} name={}".format(_input_size, _name))
    Tools.print("classier={}".format(_checkpoint_path_classier))

    _net = Classifier(input_size_or_list=[_input_size, 512, 256], low_dim=_low_dim,
                      feature_name=_feature_name, input_size=_image_size, conv1_stride=_conv1_stride,
                      is_fine_tune=_is_fine_tune, linear_bias=_linear_bias)
    runner = ClassierRunner(net=_net, max_epoch=_max_epoch, resume=False, is_fine_tune=_is_fine_tune,
                            input_size=_image_size, pre_train_path=_pre_train_path,
                            checkpoint_path=_checkpoint_path_classier)
    Tools.print()
    train_acc = runner.test(0, is_test_test=False)
    test_acc = runner.test(0, is_test_test=True)
    Tools.print('Random accuracy: {:.2f}/{:.2f}'.format(train_acc, test_acc))
    runner.train(start_epoch=_start_epoch, test_per_epoch=1)
    Tools.print()
    train_acc = runner.test(0, is_test_test=False)
    test_acc = runner.test(0, is_test_test=True)
    Tools.print('final accuracy: {:.2f}/{:.2f}'.format(train_acc, test_acc))
    pass
