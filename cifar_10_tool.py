import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.utils.data as data
import torch.nn.functional as F
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


class ImageNetInstance(datasets.ImageFolder):

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, index

    @staticmethod
    def data(train_root, test_root, batch_size=128, output_size=224, is_train_shuffle=True):
        Tools.print('==> Preparing data..')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=output_size, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])

        transform_test = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(output_size), transforms.ToTensor(), normalize])

        train_set = ImageNetInstance(root=train_root, transform=transform_train)
        test_set = ImageNetInstance(root=test_root, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=is_train_shuffle, num_workers=8)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

        class_name = None

        return train_set, train_loader, test_set, test_loader, class_name

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
    def data(data_root, batch_size=128, is_train_shuffle=True):
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
        test_set = CIFAR10Instance(root=data_root, train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   shuffle=is_train_shuffle, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

        class_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return train_set, train_loader, test_set, test_loader, class_name

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
    def data(data_root, batch_size=128, is_train_shuffle=True):
        Tools.print('==> Preparing data..')

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = CIFAR100Instance(root=data_root, train=True, download=True, transform=transform_train)
        test_set = CIFAR100Instance(root=data_root, train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   shuffle=is_train_shuffle, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

        class_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return train_set, train_loader, test_set, test_loader, class_name

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
    def data(data_root, batch_size=128, input_size=32, is_test_train_shuffle=False):
        Tools.print('==> Preparing data..')

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=input_size, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        class_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        train_set = STL10Instance(root=data_root, split="unlabeled", download=True, transform=transform_train)
        test_train_set = STL10Instance(root=data_root, split="train", download=True,
                                       transform=transform_train if is_test_train_shuffle else transform_test)
        test_test_set = STL10Instance(root=data_root, split="test", download=True, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        test_train_loader = torch.utils.data.DataLoader(test_train_set, batch_size,
                                                        shuffle=is_test_train_shuffle, num_workers=2)
        test_test_loader = torch.utils.data.DataLoader(test_test_set, batch_size, shuffle=False, num_workers=2)
        return train_set, train_loader, test_train_set, test_train_loader, test_test_set, test_test_loader, class_name

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


class KNN(object):

    @staticmethod
    def knn(epoch, net, low_dim_list, train_loader, test_loader, k, t, loader_n=1):
        net.eval()
        n_sample = train_loader.dataset.__len__()
        out_memory_list = [torch.rand(n_sample, low_dim).t().cuda() for low_dim in low_dim_list]

        targets = train_loader.dataset.train_labels if hasattr(train_loader.dataset, "train_labels") else(
            train_loader.dataset.targets if hasattr(train_loader.dataset, "targets") else train_loader.dataset.labels)

        train_labels = torch.LongTensor(targets).cuda()
        max_c = train_labels.max() + 1

        transform_bak = train_loader.dataset.transform
        train_loader.dataset.transform = test_loader.dataset.transform
        temp_loader = torch.utils.data.DataLoader(train_loader.dataset, 100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, _, indexes) in enumerate(temp_loader):
            feature_dict = net(inputs)
            batch_size = inputs.size(0)
            for _index in range(len(low_dim_list)):
                feature_name = "{}{}".format(FeatureName.L2norm, _index)
                out_memory_list[_index][:, batch_idx * batch_size:
                                           batch_idx * batch_size + batch_size] = feature_dict[feature_name].data.t()
                pass
            pass
        train_loader.dataset.transform = transform_bak

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
            now_loader = [test_loader] if loader_n == 1 else [test_loader, train_loader]

            for loader in now_loader:
                total = 0
                top1_list, top5_list = [0. for _ in low_dim_list], [0. for _ in low_dim_list]
                retrieval_one_hot_list = [torch.zeros(k, max_c).cuda() for _ in low_dim_list]  # [200, 10]

                for batch_idx, (inputs, targets, indexes) in enumerate(loader):
                    targets = targets.cuda(async=True)
                    total += targets.size(0)

                    feature_dict = net(inputs)
                    for i in range(len(low_dim_list)):
                        dist = torch.mm(feature_dict["{}{}".format(FeatureName.L2norm, i)], out_memory_list[i])
                        top1_list[i], top5_list[i], retrieval_one_hot_list[i] = _cal(
                            inputs, dist, train_labels, retrieval_one_hot_list[i], top1_list[i], top5_list[i])
                        pass
                    pass

                for i in range(len(low_dim_list)):
                    Tools.print("Test {} {} Top1={:.2f} Top5={:.2f}".format(
                        epoch, i, top1_list[i] * 100. / total, top5_list[i] * 100. / total))
                    pass

                all_acc.append(top1_list[-1] / total)
                pass
            pass

        return all_acc[0]

    pass


class FeatureName(object):

    feature = "feature"
    label = "label"

    x = "x"
    ConvB1 = "ConvB1"
    ConvB2 = "ConvB2"
    ConvB3 = "ConvB3"
    ConvB4 = "ConvB4"
    ConvB5 = "ConvB5"
    AvgPool = "AvgPool"
    L2norm = "L2norm"
    L2norm0 = "L2norm0"
    L2norm1 = "L2norm1"
    L2norm2 = "L2norm2"
    L2norm3 = "L2norm3"
    L2norm4 = "L2norm4"
    L2norm5 = "L2norm5"

    Logits = "Logits0"
    Logits0 = "Logits0"
    Logits1 = "Logits1"
    Logits2 = "Logits2"
    Logits3 = "Logits3"
    Logits4 = "Logits4"
    Logits5 = "Logits5"

    pass
