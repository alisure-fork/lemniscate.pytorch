import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optimizer
import torch.utils.data as data
from alisuretool.Tools import Tools
import torchvision.datasets as data_set
import torchvision.transforms as transforms
from cifar_9_class128_update_epoch_norm_count_3level import AttentionResNet
from cifar_9_class128_update_epoch_norm_count_3level import AttentionBasicBlock


class CIFAR10Instance(data_set.CIFAR10):

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    @staticmethod
    def data(data_root, is_train_shuffle=True):
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
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=is_train_shuffle, num_workers=2)

        test_set = CIFAR10Instance(root=data_root, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

        class_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return train_set, train_loader, test_set, test_loader, class_name

    pass


class CIFAR10DataFeature(data.Dataset):

    def __init__(self, _data, label):
        self.data = _data
        self.label = label
        pass

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        return img, target

    def __len__(self):
        return len(self.data)

    @staticmethod
    def data(train_data, train_label, test_data, test_label, is_train_shuffle=True):
        train_set = CIFAR10DataFeature(_data=train_data, label=train_label)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=is_train_shuffle, num_workers=2)

        test_set = CIFAR10DataFeature(_data=test_data, label=test_label)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

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


class Features(object):

    def __init__(self, net, features_result_path=None, checkpoint_path="./ckpt.t7", data_root='./data', low_dim=None):
        self.features_result_path = features_result_path
        self.checkpoint_path = checkpoint_path
        self.data_root = data_root
        self.low_dim = low_dim

        self.train_set, self.train_loader, self.test_set, self.test_loader, _ = CIFAR10Instance.data(
            self.data_root, is_train_shuffle=False)
        self.train_num = self.train_set.__len__()

        self.net = net
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

        self._load_model(self.net)
        pass

    def _load_model(self, net):
        if self.checkpoint_path:
            Tools.print('==> Pre train from checkpoint {} ..'.format(self.checkpoint_path))
            checkpoint = torch.load(self.checkpoint_path)
            Tools.print("epoch={} acc={}".format(checkpoint['epoch'], checkpoint['acc']))
            net.load_state_dict(checkpoint['net'], strict=True)
            pass
        pass

    def feature(self, loader):
        labels_list = []
        out_list = [[] for _ in range(len(self.low_dim) * 2)]
        for batch_idx, (inputs, targets, _) in enumerate(loader):
            if batch_idx % 100 == 0:
                Tools.print("Feature {}/{}".format(batch_idx * loader.batch_size, loader.dataset.__len__()))
            inputs = inputs.cuda()
            out = self.net(inputs)
            assert len(out) == len(out_list)

            labels_list.extend(np.asarray(targets))
            for i in range(len(out)):
                out_list[i].extend(np.asarray(out[i].detach()))
                pass
            pass

        result = {"labels": labels_list}
        for index, dim in enumerate(self.low_dim):
            result["{}_logits".format(dim)] = out_list[index * 2]
            result["{}_l2norm".format(dim)] = out_list[index * 2 + 1]
            pass

        return result

    def main(self):
        if self.features_result_path and os.path.exists(self.features_result_path):
            Tools.print("{} is exist...".format(self.features_result_path))
            result = Tools.read_from_pkl(self.features_result_path)
        else:
            Tools.print("{} not exist... now to produce...".format(self.features_result_path))
            train_data = self.feature(self.train_loader)
            test_data = self.feature(self.test_loader)
            result = {"train_data": train_data, "test_data": test_data}
            if self.features_result_path:
                Tools.print("now write to {}".format(self.features_result_path))
                Tools.write_to_pkl(Tools.new_dir(self.features_result_path), result)
                pass
            pass
        return result

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


class ClassierRunner(object):

    def __init__(self, net, _data, learning_rate=0.03, max_epoch=1000,
                 resume=False, checkpoint_path="./classier.t7", data_root='./data'):
        self.data = _data
        self.learning_rate = learning_rate
        self.checkpoint_path = Tools.new_dir(checkpoint_path)
        self.resume = resume
        self.data_root = data_root

        self.best_acc = 0

        self.train_set, self.train_loader, self.test_set, self.test_loader, _ = CIFAR10DataFeature.data(
            self.data["train_data"], self.data["train_label"],
            self.data["test_data"], self.data["test_label"], is_train_shuffle=True)
        self.train_num = self.train_set.__len__()

        self.net = net.cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

        self.criterion = nn.CrossEntropyLoss().cuda()  # define loss function
        self.optimizer = optimizer.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)

        self.max_epoch = max_epoch
        pass

    def _adjust_learning_rate(self, epoch):
        if epoch < 100:
            learning_rate = self.learning_rate
        elif epoch < 200:
            learning_rate = self.learning_rate * 0.1
        elif epoch < 300:
            learning_rate = self.learning_rate * 0.01
        else:
            learning_rate = self.learning_rate * 0.001
            pass

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
            pass
        return learning_rate

    def test(self, epoch=0, is_test_test=True):
        self.net.eval()
        total = 0
        correct = 0
        loader = self.test_loader if is_test_test else self.train_loader
        for batch_idx, (inputs, targets) in enumerate(loader):
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
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer.zero_grad()

            out_logits = self.net(inputs)

            loss = self.criterion(out_logits, targets)
            avg_loss.update(loss.item(), inputs.size(0))
            loss.backward()
            self.optimizer.step()
            pass

        Tools.print('Epoch: [{}/{}] Loss: {avg_loss.val:.4f} ({avg_loss.avg:.4f})'.format(
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    """
    """

    _name = "9_class_2048_norm_count_3level_512_128_lr_1000"

    _low_dim = [2048, 512, 128]
    _features_result_path = "./checkpoint/{}/data.pkl".format(_name)
    _checkpoint_path = "./checkpoint/{}/ckpt.t7".format(_name)

    _net = AttentionResNet(AttentionBasicBlock, [2, 2, 2, 2],  *_low_dim).cuda()
    features = Features(net=_net, features_result_path=_features_result_path,
                        checkpoint_path=_checkpoint_path, data_root='./data', low_dim=_low_dim)
    feature = features.main()

    _which = 1
    _is_l2norm = False
    _start_epoch = 0
    _max_epoch = 500
    _input_size = _low_dim[_which]
    _checkpoint_path_classier = "./checkpoint/{}/classier_l_{}_{}.t7".format(_name, _input_size,
                                                                             0 if _is_l2norm else 1)

    Tools.print()
    Tools.print("input_size={} name={} checkpoint_path_classier={}".format(
        _input_size, _name, _checkpoint_path_classier))

    _net = LinearClassifier(input_size=_input_size)  # 81.62/81.17/80.63
    # _net = MultipleLinearClassifiers(input_size_list=[_input_size, 512, 256])
    # _net = MultipleNonLinearClassifier(input_size_list=[_input_size, 512, 256])

    _data = {"train_data": feature["train_data"]["{}_{}".format(_low_dim[_which],
                                                                "l2norm" if _is_l2norm else "logits")],
             "train_label": feature["train_data"]["labels"],
             "test_data": feature["test_data"]["{}_{}".format(_low_dim[_which],
                                                              "l2norm" if _is_l2norm else "logits")],
             "test_label": feature["test_data"]["labels"]}

    runner = ClassierRunner(net=_net, _data=_data, max_epoch=_max_epoch,
                            resume=False, checkpoint_path=_checkpoint_path_classier)

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
