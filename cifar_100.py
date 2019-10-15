import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from alisuretool.Tools import Tools
import torchvision.transforms as transforms
from hc_net_cifar import HCResNet as AttentionResNet
from hc_net_cifar import HCBasicBlock as AttentionBasicBlock
from hc_net_cifar100_test import MultipleNonLinearClassifier


class Classifier(nn.Module):

    def __init__(self, low_dim, input_size_or_list, output_size=10, which_out=[0, 0]):
        super(Classifier, self).__init__()
        assert len(low_dim) > which_out[0] and (which_out[1] == 1 or which_out[1] == 0)

        self.which_out = which_out

        self.attention = AttentionResNet(AttentionBasicBlock, [2, 2, 2, 2],
                                         low_dim_list=low_dim, linear_bias=False).cuda()

        assert isinstance(input_size_or_list, list) and len(input_size_or_list) >=1
        self.linear = MultipleNonLinearClassifier(input_size_or_list, output_size)
        pass

    def forward(self, inputs):
        out = self.attention(inputs)
        out = out[self.which_out[1]][self.which_out[0]]
        out = self.linear(out)
        return out

    pass


class Runner(object):

    def __init__(self, root_path='/home/ubuntu/data1.5TB/cifar', num_classes=100,
                 model=None, batch_size=128, lr=0.1, name="vgg"):
        self.root_path = root_path
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.name = name
        self.checkpoint_path = "./checkpoint/{}".format(self.name)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.best_acc = 0
        self.start_epoch = 0

        self.net = torch.nn.DataParallel(model.cuda())
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

        self.train_loader, self.test_loader = self._data()
        pass

    def _data(self):
        Tools.print('==> Preparing data..')
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_set = torchvision.datasets.CIFAR100(self.root_path, train=True, download=True, transform=transform_train)
        _train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

        test_set = torchvision.datasets.CIFAR100(self.root_path, train=False, download=True, transform=transform_test)
        _test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

        return _train_loader, _test_loader

    def _change_lr(self, epoch):

        def __change_lr(_lr):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = _lr
            pass

        if 0 <= epoch < 100:
            __change_lr(self.lr)
        elif 100 <= epoch < 200:
            __change_lr(self.lr / 10)
        elif 200 <= epoch:
            __change_lr(self.lr / 100)

        pass

    def train(self, epoch, change_lr=False):
        print()
        Tools.print('Epoch: %d' % epoch)

        if change_lr:
            self._change_lr(epoch)

        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pass
        Tools.print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                   % (train_loss / len(self.train_loader), 100. * correct / total, correct, total))
        pass

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                pass
            pass

        Tools.print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                   % (test_loss / len(self.test_loader), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > self.best_acc:
            Tools.print('Saving..')
            state = {'net': self.net.state_dict(), 'acc': acc, 'epoch': epoch}
            if not os.path.isdir(self.checkpoint_path):
                os.mkdir(self.checkpoint_path)
            torch.save(state, '{}/ckpt.t7'.format(self.checkpoint_path))
            self.best_acc = acc
            pass
        Tools.print("best_acc={} acc={}".format(self.best_acc, acc))
        pass

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 1

    # 75.82, 75.34

    _num_classes = 100
    _low_dim = [1024, 512, 256, 128, 64]
    # _low_dim = [100]
    _which = 0
    _is_l2norm = False
    _input_size = _low_dim[_which]  # first input size
    _which_out = [_which, (1 if _is_l2norm else 0)]
    _model = Classifier(input_size_or_list=[_input_size, 512, 512],
                        low_dim=_low_dim, output_size=_num_classes, which_out=_which_out)

    runner = Runner(root_path="./data", num_classes=_num_classes, model=_model,
                    batch_size=128, lr=0.01, name="ResNet18")

    for _epoch in range(runner.start_epoch, 300):
        runner.train(_epoch, change_lr=True)
        runner.test(_epoch)
        pass

    pass
