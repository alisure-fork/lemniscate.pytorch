import os
import math
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from alisuretool.Tools import Tools
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
        pass

    def forward(self, x, dim=1):
        norm = x.pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

    pass


class HCBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(HCBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(self.expansion * planes))
            pass
        pass

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    pass


class HCResNet(nn.Module):

    def __init__(self, block, num_blocks, low_dim_list, linear_bias=True):
        super(HCResNet, self).__init__()
        self.in_planes = 64
        self.low_dim_list = low_dim_list
        self.linear_bias = linear_bias

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        if len(self.low_dim_list) == 1:
            self.linear_1 = self._make_linear(512 * block.expansion)[0]
        elif len(self.low_dim_list) == 2:
            self.linear_1, self.linear_2 = self._make_linear(512 * block.expansion)
        elif len(self.low_dim_list) == 3:
            self.linear_1, self.linear_2, self.linear_3 = self._make_linear(512 * block.expansion)
        elif len(self.low_dim_list) == 4:
            self.linear_1, self.linear_2, self.linear_3, self.linear_4 = self._make_linear(512 * block.expansion)
        elif len(self.low_dim_list) == 5:
            (self.linear_1, self.linear_2, self.linear_3,
             self.linear_4, self.linear_5) = self._make_linear(512 * block.expansion)
        elif len(self.low_dim_list) == 6:
            (self.linear_1, self.linear_2, self.linear_3,
             self.linear_4, self.linear_5, self.linear_6) = self._make_linear(512 * block.expansion)
        else:
            raise Exception("low_dim_list={}".format(self.low_dim_list))

        self.l2norm = Normalize(2)
        pass

    def _make_linear(self, in_dim):
        in_dim = in_dim
        linear_list = []
        for low_dim in self.low_dim_list:
            out_dim = low_dim
            linear = nn.Linear(in_dim, out_dim, bias=self.linear_bias)
            in_dim = out_dim
            linear_list.append(linear)
            pass
        return linear_list

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        logits, l2norm = [], []
        if len(self.low_dim_list) >= 1:
            out_logits = self.linear_1(out)
            out_l2norm = self.l2norm(out_logits)
            logits.append(out_logits)
            l2norm.append(out_l2norm)
            if len(self.low_dim_list) >= 2:
                out_logits2 = self.linear_2(out_logits)
                out_l2norm2 = self.l2norm(out_logits2)
                logits.append(out_logits2)
                l2norm.append(out_l2norm2)
                if len(self.low_dim_list) >= 3:
                    out_logits3 = self.linear_3(out_logits2)
                    out_l2norm3 = self.l2norm(out_logits3)
                    logits.append(out_logits3)
                    l2norm.append(out_l2norm3)
                    if len(self.low_dim_list) >= 4:
                        out_logits4 = self.linear_4(out_logits3)
                        out_l2norm4 = self.l2norm(out_logits4)
                        logits.append(out_logits4)
                        l2norm.append(out_l2norm4)
                        if len(self.low_dim_list) >= 5:
                            out_logits5 = self.linear_5(out_logits4)
                            out_l2norm5 = self.l2norm(out_logits5)
                            logits.append(out_logits5)
                            l2norm.append(out_l2norm5)
                            if len(self.low_dim_list) >= 6:
                                out_logits6 = self.linear_6(out_logits5)
                                out_l2norm6 = self.l2norm(out_logits6)
                                logits.append(out_logits6)
                                l2norm.append(out_l2norm6)
                                pass
                            pass
                        pass
                    pass
                pass
            pass

        return logits, l2norm

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


class KNN(object):

    @staticmethod
    def knn(epoch, net, low_dim_list, test_train_loader, test_test_loader, k, t, loader_n=1):
        net.eval()
        n_sample = test_train_loader.dataset.__len__()
        out_memory_list = [torch.rand(n_sample, low_dim).t().cuda() for low_dim in low_dim_list]

        _d = test_train_loader.dataset
        train_labels = _d.train_labels if hasattr(_d, "train_labels") else(
            _d.targets if hasattr(_d, "targets") else _d.labels)
        train_labels = torch.LongTensor(train_labels).cuda()

        max_c = train_labels.max() + 1
        for batch_idx, (inputs, _, indexes) in enumerate(test_train_loader):
            logits, l2norm = net(inputs)
            batch_size = inputs.size(0)
            for i in range(len(low_dim_list)):
                out_memory_list[i][:, batch_idx * batch_size:batch_idx * batch_size + batch_size] = l2norm[i].data.t()
            pass

        def _cal(inputs, dist, train_labels, retrieval_one_hot, top1, top5):
            # ---------------------------------------------------------------------------------- #
            batch_size = inputs.size(0)
            yd, yi = dist.topk(k, dim=1, largest=True, sorted=True)
            candidates = train_labels.view(1, -1).expand(batch_size, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batch_size * k, max_c).zero_()
            retrieval_one_hot = retrieval_one_hot.scatter_(1, retrieval.view(-1, 1),
                                                           1).view(batch_size, -1, max_c)
            yd_transform = yd.clone().div_(t).exp_().view(batch_size, -1, 1)
            probs = torch.sum(torch.mul(retrieval_one_hot, yd_transform), 1)
            _, predictions = probs.sort(1, True)
            # ---------------------------------------------------------------------------------- #

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1, 1))

            top1 += correct.narrow(1, 0, 1).sum().item()
            top5 += correct.narrow(1, 0, 5).sum().item()
            return top1, top5, retrieval_one_hot

        all_acc = []
        with torch.no_grad():
            now_loader = [test_test_loader] if loader_n == 1 else [test_test_loader, test_train_loader]
            for loader in now_loader:
                total = 0
                top1_list = [0. for _ in range(len(low_dim_list))]
                top5_list = [0. for _ in range(len(low_dim_list))]
                retrieval_list = [torch.zeros(k, max_c).cuda() for _ in range(len(low_dim_list))]
                for batch_idx, (inputs, targets, indexes) in enumerate(loader):
                    targets = targets.cuda(async=True)
                    total += targets.size(0)
                    logits, l2norm = net(inputs)
                    for i in range(len(low_dim_list)):
                        dist = torch.mm(l2norm[i], out_memory_list[i])
                        top1_list[i], top5_list[i], retrieval_list[i] = _cal(
                            inputs, dist, train_labels, retrieval_list[i], top1_list[i], top5_list[i])
                        pass
                    pass

                # now = "Test {}".format(epoch)
                # for i in range(len(low_dim_list)):
                #     now += " {} Top1={:.2f} Top5={:.2f},".format(i, top1_list[i] * 100. / total,
                #                                                  top5_list[i] * 100. / total)
                #     pass
                # Tools.print(now)

                Tools.print("Test {}".format(epoch))
                for i in range(len(low_dim_list)):
                    Tools.print(" {} Top1={:.2f} Top5={:.2f},".format(i, top1_list[i] * 100. / total,
                                                                      top5_list[i] * 100. / total))
                    pass

                all_acc.append(top1_list[-1] / total)
                pass
            pass

        return all_acc[0]

    pass


class HCLoss(nn.Module):

    def __init__(self):
        super(HCLoss, self).__init__()
        self.criterion_no = nn.CrossEntropyLoss()
        pass

    def forward(self, out, targets, param, l1_lambda=0.1):
        loss_1 = self.criterion_no(out, targets)
        loss_2 = torch.norm(param, 1) / out.size()[-1]
        return loss_1 + l1_lambda * loss_2, loss_1, loss_2

    pass


class ProduceClass(object):

    def __init__(self, n_sample, low_dim, ratio=1.0):
        super(ProduceClass, self).__init__()
        self.low_dim = low_dim
        self.n_sample = n_sample
        self.class_per_num = self.n_sample // self.low_dim * ratio
        self.count = 0
        self.count_2 = 0
        self.class_num = np.zeros(shape=(self.low_dim, ), dtype=np.int)
        self.classes = np.zeros(shape=(self.n_sample, ), dtype=np.int)
        pass

    def reset(self):
        self.count = 0
        self.count_2 = 0
        self.class_num *= 0
        pass

    def cal_label(self, out, indexes):
        top_k = out.data.topk(self.low_dim, dim=1)[1].cpu()
        indexes_cpu = indexes.cpu()

        batch_size = top_k.size(0)
        class_labels = np.zeros(shape=(batch_size,), dtype=np.int)

        for i in range(batch_size):
            for j_index, j in enumerate(top_k[i]):
                if self.class_per_num > self.class_num[j]:
                    class_labels[i] = j
                    self.class_num[j] += 1
                    self.count += 1 if self.classes[indexes_cpu[i]] != j else 0
                    self.classes[indexes_cpu[i]] = j
                    self.count_2 += 1 if j_index != 0 else 0
                    break
                pass
            pass
        pass

    def get_label(self, indexes):
        return torch.tensor(self.classes[indexes.cpu()]).long().cuda()

    pass


class CIFAR10Instance(datasets.CIFAR10):

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
    def data(data_root, batch_size=128):
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

        class_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        train_set = CIFAR10Instance(root=data_root, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

        test_train_set = CIFAR10Instance(root=data_root, train=True, download=True, transform=transform_test)
        test_train_loader = torch.utils.data.DataLoader(test_train_set, batch_size, shuffle=False, num_workers=2)

        test_test_set = CIFAR10Instance(root=data_root, train=False, download=True, transform=transform_test)
        test_test_loader = torch.utils.data.DataLoader(test_test_set, batch_size, shuffle=False, num_workers=2)

        return train_set, train_loader, test_train_set, test_train_loader, test_test_set, test_test_loader, class_name

    pass


class CIFAR100Instance(datasets.CIFAR100):

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
    def data(data_root, batch_size=128):
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

        class_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        train_set = CIFAR100Instance(root=data_root, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

        test_train_set = CIFAR100Instance(root=data_root, train=True, download=True, transform=transform_test)
        test_train_loader = torch.utils.data.DataLoader(test_train_set, batch_size, shuffle=False, num_workers=2)

        test_test_set = CIFAR100Instance(root=data_root, train=False, download=True, transform=transform_test)
        test_test_loader = torch.utils.data.DataLoader(test_test_set, batch_size, shuffle=False, num_workers=2)

        return train_set, train_loader, test_train_set, test_train_loader, test_test_set, test_test_loader, class_name

    pass


class STL10Instance(datasets.STL10):

    def __getitem__(self, index):
        if hasattr(self, "data") and hasattr(self, "labels"):
            if self.labels is not None:
                img, target = self.data[index], int(self.labels[index])
            else:
                img, target = self.data[index], None
        else:
            if self.train:
                img, target = self.train_data[index], self.train_labels[index]
            else:
                img, target = self.test_data[index], self.test_labels[index]
            pass

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    @staticmethod
    def data(data_root, batch_size=128):
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
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        class_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        train_set = STL10Instance(root=data_root, split="unlabeled", download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

        test_train_set = STL10Instance(root=data_root, split="train", download=True, transform=transform_test)
        test_train_loader = torch.utils.data.DataLoader(test_train_set, batch_size, shuffle=False, num_workers=2)

        test_test_set = STL10Instance(root=data_root, split="test", download=True, transform=transform_test)
        test_test_loader = torch.utils.data.DataLoader(test_test_set, batch_size, shuffle=False, num_workers=2)

        return train_set, train_loader, test_train_set, test_train_loader, test_test_set, test_test_loader, class_name

    pass


class DataSet(object):

    def __init__(self, data_set_name, data_root, batch_size):
        self.data_set_name = data_set_name
        self.data_set_name_list = ["cifar10", "cifar100", "stl10"]
        self.data_set_instance_list = [CIFAR10Instance, CIFAR100Instance, STL10Instance]
        assert self.data_set_name in self.data_set_name_list
        self.data_root = data_root
        self.batch_size = batch_size
        (self.train_set, self.train_loader, self.test_train_set, self.test_train_loader,
         self.test_test_set, self.test_test_loader, self.class_name) = self.produce_data()
        pass

    def produce_data(self):
        _instance = None
        for name, instance in zip(self.data_set_name_list, self.data_set_instance_list):
            if self.data_set_name == name:
                _instance = instance
                break
            pass
        assert _instance is not None

        (train_set, train_loader, test_train_set, test_train_loader, test_test_set,
         test_test_loader, class_name) = _instance.data(self.data_root, batch_size=self.batch_size)
        return train_set, train_loader, test_train_set, test_train_loader, test_test_set, test_test_loader, class_name

    def data(self):
        return (self.train_set, self.train_loader, self.test_train_set,
                self.test_train_loader,  self.test_test_set, self.test_test_loader)

    pass


class HCRunner(object):

    def __init__(self, low_dim_list, ratio_list, data_set_name="stl10", batch_size=128, is_adjust_lambda=False,
                 l1_lambda=0.1, learning_rate=0.03, linear_bias=True, has_l1=False, max_epoch=1000, t_epoch=300,
                 first_epoch=200, resume=False, checkpoint_path="./ckpt.t7", pre_train=None, data_root='./data'):
        self.learning_rate = learning_rate
        self.checkpoint_path = Tools.new_dir(checkpoint_path)
        self.resume = resume
        self.pre_train = pre_train
        self.data_root = data_root
        self.batch_size = batch_size
        self.data_set_name = data_set_name

        self.low_dim_list = low_dim_list
        self.ratio_list = ratio_list

        self.t_epoch = t_epoch
        self.max_epoch = max_epoch
        self.first_epoch = first_epoch
        self.linear_bias = linear_bias
        self.has_l1 = has_l1
        self.l1_lambda = l1_lambda
        self.is_adjust_lambda = is_adjust_lambda

        self.best_acc = 0

        (self.train_set, self.train_loader, self.test_train_set, self.test_train_loader, self.test_test_set,
         self.test_test_loader) = DataSet(self.data_set_name, self.data_root, batch_size=self.batch_size).data()
        self.train_num = self.train_set.__len__()

        self.net = HCResNet(HCBasicBlock, [2, 2, 2, 2], self.low_dim_list, linear_bias=linear_bias).cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

        self._load_model(self.net)

        self.produce_class_list = [ProduceClass(n_sample=self.train_num, low_dim=self.low_dim_list[i],
                                                ratio=self.ratio_list[i]) for i in range(len(self.low_dim_list))]
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
                [self.produce_class_list[i].reset() for i in range(len(self.low_dim_list))]
                for batch_idx, (inputs, _, indexes) in enumerate(self.train_loader):
                    inputs, indexes = inputs.cuda(), indexes.cuda()
                    out_logits, out_l2norm = self.net(inputs)
                    [self.produce_class_list[i].cal_label(out_l2norm[i],
                                                          indexes) for i in range(len(self.low_dim_list))]
                    pass
                now = "Epoch: [{}]".format(epoch)
                for i in range(len(self.low_dim_list)):
                    now += " {}-{}/{}".format(i+1, self.produce_class_list[i].count,
                                              self.produce_class_list[i].count_2)
                    pass
                Tools.print(now)
                pass
        finally:
            pass

        # Train
        try:
            self.net.train()
            _learning_rate_ = self._adjust_learning_rate(epoch)
            _l1_lambda_ = self._adjust_l1_lambda(epoch)
            Tools.print('Epoch: {} lr={} lambda={}'.format(epoch, _learning_rate_, _l1_lambda_))
            avg_loss_list = [AverageMeter() for _ in range(len(self.low_dim_list))]
            avg_loss_1_list = [AverageMeter() for _ in range(len(self.low_dim_list))]
            avg_loss_2_list = [AverageMeter() for _ in range(len(self.low_dim_list))]

            for batch_idx, (inputs, _, indexes) in enumerate(self.train_loader):
                inputs, indexes = inputs.cuda(), indexes.cuda()
                self.optimizer.zero_grad()

                out_logits, out_l2norm = self.net(inputs)
                targets_list = [self.produce_class_list[i].get_label(indexes) for i in range(len(self.low_dim_list))]

                params = [_ for _ in self.net.module.parameters()]
                loss_list, loss_1_list, loss_2_list = [], [], []
                for i in range(len(self.low_dim_list)):
                    loss_1, loss_1_1, loss_1_2 = self.criterion(out_logits[i], targets_list[i],
                                                                params[i-len(self.low_dim_list)], _l1_lambda_)
                    loss_list.append(loss_1)
                    loss_1_list.append(loss_1_1)
                    loss_2_list.append(loss_1_2)
                    avg_loss_list[i].update(loss_1.item(), inputs.size(0))
                    avg_loss_1_list[i].update(loss_1_1.item(), inputs.size(0))
                    avg_loss_2_list[i].update(loss_1_2.item(), inputs.size(0))
                    pass

                loss = np.sum(loss_list) if self.has_l1 else np.sum(loss_1_list)
                loss.backward()

                self.optimizer.step()
                pass

            now = "Epoch: [{}/{}]".format(epoch, len(self.train_loader))
            for i in range(len(self.low_dim_list)):
                now += " Loss {}: {:.2f}({:.2f}/{:.2f})".format(i + 1, avg_loss_list[i].avg,
                                                                avg_loss_1_list[i].avg, avg_loss_2_list[i].avg)
                pass
            Tools.print(now)
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
    # 11_class_1024_512_256_128_64_no_1600_32_1_l1_sum_0_54321
    87.73(1024) 87.91(512) 87.95(256) 87.69(128) 87.69(64)
    """

    _hc_num = 5
    if _hc_num == 1:
        _low_dim_list = [64]
        _ratio_list = [1]
    elif _hc_num == 2:
        _low_dim_list = [128, 64]
        _ratio_list = [2, 1]
    elif _hc_num == 3:
        _low_dim_list = [256, 128, 64]
        _ratio_list = [3, 2, 1]
    elif _hc_num == 4:
        _low_dim_list = [512, 256, 128, 64]
        _ratio_list = [4, 3, 2, 1]
    elif _hc_num == 5:
        _low_dim_list = [1024, 512, 256, 128, 64]
        _ratio_list = [5, 4, 3, 2, 1]
    else:
        raise Exception("...hc_num={}...".format(_hc_num))

    _data_set_name = "cifar10"  # cifar10, cifar100, stl10
    _start_epoch, _max_epoch = 0, 1600
    _learning_rate = 0.01
    _first_epoch, _t_epoch = 200, 100
    _l1_lambda, _is_adjust_lambda = 0.0, False

    _data_root = './data'
    _batch_size = 32
    _has_l1 = True
    _linear_bias = False
    _resume = False
    _pre_train = None
    # pre_train = "./checkpoint/。。。/ckpt.t7"

    _low_dim_list_str = "".join(["_{}".format(low_dim) for low_dim in _low_dim_list])
    _ratio_list_str = "".join(["{}".format(ratio) for ratio in _ratio_list])
    _name = "hc_{}_{}_dim{}_ratio_{}_epoch_{}_bz_{}_bias_{}_lambda_{}_sum".format(
        _data_set_name, _hc_num, _low_dim_list_str, _ratio_list_str, _max_epoch, _batch_size,
        0 if _linear_bias else 1, 1 if _is_adjust_lambda else 0)
    _checkpoint_path = "./checkpoint/{}/ckpt.t7".format(_name)

    Tools.print()
    Tools.print("name={}".format(_name))
    Tools.print("low_dim_list={}".format(_low_dim_list))
    Tools.print("ratio_list={}".format(_ratio_list))
    Tools.print("learning_rate={} batch_size={}".format(_learning_rate, _batch_size))
    Tools.print("has_l1={} l1_lambda={} is_adjust_lambda={}".format(_has_l1, _l1_lambda, _is_adjust_lambda))
    Tools.print("pre_train={} checkpoint_path={}".format(_pre_train, _checkpoint_path))
    Tools.print()

    runner = HCRunner(low_dim_list=_low_dim_list, ratio_list=_ratio_list, linear_bias=_linear_bias,
                      has_l1=_has_l1, data_set_name=_data_set_name, l1_lambda=_l1_lambda,
                      is_adjust_lambda=_is_adjust_lambda, batch_size=_batch_size, learning_rate=_learning_rate,
                      max_epoch=_max_epoch, t_epoch=_t_epoch, first_epoch=_first_epoch, resume=_resume,
                      pre_train=_pre_train, checkpoint_path=_checkpoint_path, data_root=_data_root)
    Tools.print()
    acc = runner.test()
    Tools.print('Random accuracy: {:.2f}'.format(acc * 100))
    runner.train(start_epoch=_start_epoch, update_epoch=1)
    Tools.print()
    acc = runner.test(loader_n=2)
    Tools.print('final accuracy: {:.2f}'.format(acc * 100))
    pass
