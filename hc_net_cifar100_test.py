import os
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optimizer
from alisuretool.Tools import Tools
import torchvision.datasets as data_set
import torchvision.transforms as transforms
from hc_net_cifar import HCBasicBlock as AttentionBasicBlock


class CIFAR100Instance(data_set.CIFAR100):

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
    def data(data_root, is_train_shuffle=True, batch_size=128):
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

        train_set = CIFAR100Instance(root=data_root, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   shuffle=is_train_shuffle, num_workers=2)

        test_set = CIFAR100Instance(root=data_root, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

        class_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return train_set, train_loader, test_set, test_loader, class_name

    pass


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        pass

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        pass

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        pass

    pass


class LinearClassifier(nn.Module):

    def __init__(self, input_size, output_size=10):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        pass

    def forward(self, inputs):
        return self.linear(inputs)

    pass


class MultipleLinearClassifiers(nn.Module):

    def __init__(self, input_size_list, output_size=10):
        super(MultipleLinearClassifiers, self).__init__()
        self.linear_1 = LinearClassifier(input_size_list[0], input_size_list[1])
        self.linear_2 = LinearClassifier(input_size_list[1], input_size_list[2])
        self.linear_3 = LinearClassifier(input_size_list[2], output_size)
        pass

    def forward(self, inputs):
        return self.linear_3(self.linear_2(self.linear_1(inputs)))

    pass


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

    def __init__(self, low_dim, input_size_or_list, output_size=10,
                 classifier_type=0, which_out=[0, 0], is_fine_tune=False, linear_bias=True):
        super(Classifier, self).__init__()
        assert len(low_dim) > which_out[0] and (which_out[1] == 1 or which_out[1] == 0)

        self.which_out = which_out
        self.is_fine_tune = is_fine_tune

        self.attention = AttentionResNet(AttentionBasicBlock, [2, 2, 2, 2],
                                         low_dim_list=low_dim, linear_bias=linear_bias).cuda()

        if not self.is_fine_tune:
            for p in self.parameters():
                p.requires_grad = False
                pass
            pass

        if classifier_type == 1:
            assert isinstance(input_size_or_list, list) and len(input_size_or_list) >=1
            self.linear = MultipleLinearClassifiers(input_size_or_list, output_size)
        elif classifier_type == 2:
            assert isinstance(input_size_or_list, list) and len(input_size_or_list) >=1
            self.linear = MultipleNonLinearClassifier(input_size_or_list, output_size)
        else:
            assert isinstance(input_size_or_list, int)
            self.linear = LinearClassifier(input_size_or_list, output_size)
        pass

    def forward(self, inputs):
        out = self.attention(inputs)
        out = out[self.which_out[1]][self.which_out[0]]
        out = self.linear(out)
        return out
    pass


class ClassierRunner(object):

    def __init__(self, net, learning_rate=0.03, fine_tune_learning_rate=0.001,
                 max_epoch=1000, resume=False, is_fine_tune=False,
                 pre_train_path=None, checkpoint_path="./classier.t7", data_root='./data'):
        self.learning_rate = learning_rate
        self.checkpoint_path = Tools.new_dir(checkpoint_path)
        self.pre_train_path = pre_train_path
        self.resume = resume
        self.is_fine_tune = is_fine_tune
        self.data_root = data_root

        self.best_acc = 0

        if self.is_fine_tune:
            self.learning_rate = fine_tune_learning_rate
            pass

        self.train_set, self.train_loader, self.test_set, self.test_loader, _ = CIFAR100Instance.data(
            self.data_root, is_train_shuffle=True, batch_size=64)
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
        loader = self.test_loader if is_test_test else self.train_loader
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
        for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer.zero_grad()

            out_logits = self.net(inputs)

            loss = self.criterion(out_logits, targets)
            avg_loss.update(loss.item(), inputs.size(0))
            loss.backward()
            self.optimizer.step()
            pass

        Tools.print('Epoch: [{}-{}] Loss: {avg_loss.val:.4f} ({avg_loss.avg:.4f})'.format(
            epoch, len(self.train_loader), avg_loss=avg_loss))

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    """
    # 0.5247 1393 hc_cifar100_5_dim_1024_512_256_128_64_ratio_54321_epoch_1600_bz_32_bias_1_lambda_0_sum
    2: 0.6144 classier_1024_2_0_0
    2: 0.7118 classier_1024_2_0_0 fine tune
    2: 0.6618 classier_1024_2_0_0 init
    
    2: 0.6029 classier_64_2_4_0
    2: 0.7110 classier_64_2_4_0 fine tune
    2: 0.xxxx classier_64_2_4_0 init
    """

    # 1
    _low_dim = [1024, 512, 256, 128, 64]
    _name = "hc_cifar100_5_dim_1024_512_256_128_64_ratio_54321_epoch_1600_bz_32_bias_1_lambda_0_sum"
    from hc_net_cifar import HCResNet as AttentionResNet

    _which = 0
    _is_l2norm = False
    _classifier_type = 2  # 0, 1, 2
    _learning_rate = 0.01
    _is_fine_tune = True
    _fine_tune_learning_rate = 0.01
    # _pre_train_path = None
    _pre_train_path = "./checkpoint/{}/ckpt.t7".format(_name)

    _which_out = [_which, (1 if _is_l2norm else 0)]
    _input_size = _low_dim[_which]  # first input size

    _start_epoch = 0  # train epoch
    _max_epoch = 200
    _linear_bias = False

    _checkpoint_path_classier = "./checkpoint/{}/classier_{}_{}_{}_{}_{}.t7".format(
        _name, _input_size, _classifier_type, _which_out[0], _which_out[1], 1 if _is_fine_tune else 0)

    Tools.print()
    Tools.print("input_size={} name={}".format(_input_size, _name))
    Tools.print("classier={}".format(_checkpoint_path_classier))

    _net = Classifier(input_size_or_list=_input_size if _classifier_type == 0 else [_input_size, 512, 256],
                      low_dim=_low_dim, classifier_type=_classifier_type, output_size=100,
                      which_out=_which_out, is_fine_tune=_is_fine_tune, linear_bias=_linear_bias)
    runner = ClassierRunner(net=_net, learning_rate=_learning_rate, fine_tune_learning_rate=_fine_tune_learning_rate,
                            max_epoch=_max_epoch, resume=False, is_fine_tune=_is_fine_tune,
                            pre_train_path=_pre_train_path, checkpoint_path=_checkpoint_path_classier)

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
