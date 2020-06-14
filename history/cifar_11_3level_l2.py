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


class AttentionBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(AttentionBasicBlock, self).__init__()
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


class AttentionResNet(nn.Module):

    def __init__(self, block, num_blocks, low_dim=512, low_dim2=128, low_dim3=10, linear_bias=True):
        super(AttentionResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear_512 = nn.Linear(512 * block.expansion, low_dim, bias=linear_bias)
        self.linear_128 = nn.Linear(low_dim, low_dim2, bias=linear_bias)
        self.linear_10 = nn.Linear(low_dim2, low_dim3, bias=linear_bias)
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out_logits = self.linear_512(out)
        out_l2norm = self.l2norm(out_logits)

        out_logits2 = self.linear_128(out_logits)
        out_l2norm2 = self.l2norm(out_logits2)

        out_logits3 = self.linear_10(out_logits2)
        out_l2norm3 = self.l2norm(out_logits3)
        return out_logits, out_l2norm, out_logits2, out_l2norm2, out_logits3, out_l2norm3

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


class CIFAR10Instance(datasets.CIFAR10):

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
    def data(data_root):
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
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers=2)

        test_set = CIFAR10Instance(root=data_root, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

        class_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return train_set, train_loader, test_set, test_loader, class_name

    pass


class KNN(object):

    @staticmethod
    def knn(epoch, net, produce_class, produce_class2, produce_class3,
            train_loader, test_loader, k, t, recompute_memory=0, loader_n=1):
        net.eval()

        out_memory = produce_class.memory.t()
        out_memory2 = produce_class2.memory.t()
        out_memory3 = produce_class3.memory.t()
        train_labels = torch.LongTensor(train_loader.dataset.train_labels).cuda()
        c = train_labels.max() + 1

        if recompute_memory:
            transform_bak = train_loader.dataset.transform

            train_loader.dataset.transform = test_loader.dataset.transform
            temp_loader = torch.utils.data.DataLoader(train_loader.dataset, 100, shuffle=False, num_workers=1)
            for batch_idx, (inputs, _, indexes) in enumerate(temp_loader):
                out_logits, out_l2norm, out_logits2, out_l2norm2, out_logits3, out_l2norm3 = net(inputs)
                batch_size = inputs.size(0)
                out_memory[:, batch_idx * batch_size:batch_idx * batch_size + batch_size] = out_l2norm.data.t()
                out_memory2[:, batch_idx * batch_size:batch_idx * batch_size + batch_size] = out_l2norm2.data.t()
                out_memory3[:, batch_idx * batch_size:batch_idx * batch_size + batch_size] = out_l2norm3.data.t()
                pass

            train_loader.dataset.transform = transform_bak
            pass

        def _cal(inputs, dist, train_labels, retrieval_one_hot, top1, top5):
            # ---------------------------------------------------------------------------------- #
            batch_size = inputs.size(0)
            yd, yi = dist.topk(k, dim=1, largest=True, sorted=True)
            candidates = train_labels.view(1, -1).expand(batch_size, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batch_size * k, c).zero_()
            retrieval_one_hot = retrieval_one_hot.scatter_(1, retrieval.view(-1, 1),
                                                           1).view(batch_size, -1, c)
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
                top1, top5 = 0., 0.
                top12, top52 = 0., 0.
                top13, top53 = 0., 0.
                total = 0

                retrieval_one_hot = torch.zeros(k, c).cuda()  # [200, 10]
                retrieval_one_hot2 = torch.zeros(k, c).cuda()  # [200, 10]
                retrieval_one_hot3 = torch.zeros(k, c).cuda()  # [200, 10]
                for batch_idx, (inputs, targets, indexes) in enumerate(loader):
                    targets = targets.cuda(async=True)
                    total += targets.size(0)

                    out_logits, out_l2norm, out_logits2, out_l2norm2, out_logits3, out_l2norm3 = net(inputs)
                    dist = torch.mm(out_l2norm, out_memory)
                    dist2 = torch.mm(out_l2norm2, out_memory2)
                    dist3 = torch.mm(out_l2norm3, out_memory3)
                    top1, top5, retrieval_one_hot = _cal(inputs, dist, train_labels, retrieval_one_hot, top1, top5)
                    top12, top52, retrieval_one_hot2 = _cal(inputs, dist2, train_labels,
                                                            retrieval_one_hot2, top12, top52)
                    top13, top53, retrieval_one_hot3 = _cal(inputs, dist3, train_labels,
                                                            retrieval_one_hot3, top13, top53)
                    pass

                Tools.print("Test 1 {} Top1={:.2f} Top5={:.2f}".format(epoch, top1 * 100. / total,
                                                                       top5 * 100. / total))
                Tools.print("Test 2 {} Top1={:.2f} Top5={:.2f}".format(epoch, top12 * 100. / total,
                                                                       top52 * 100. / total))
                Tools.print("Test 3 {} Top1={:.2f} Top5={:.2f}".format(epoch, top13 * 100. / total,
                                                                       top53 * 100. / total))
                all_acc.append(top13 / total)

                pass
            pass

        return all_acc[0]

    pass


class AttentionLoss(nn.Module):

    def __init__(self):
        super(AttentionLoss, self).__init__()
        self.criterion_no = nn.CrossEntropyLoss()
        pass

    def forward(self, out, targets, param):
        loss_1 = self.criterion_no(out, targets)
        loss_2 = torch.norm(param, 1) / out.size()[-1]
        return loss_1 + loss_2, loss_1, loss_2

    pass


class ProduceClass(nn.Module):

    def __init__(self, n_sample, low_dim, momentum=0.5, ratio=1.0):
        super(ProduceClass, self).__init__()
        self.low_dim = low_dim
        self.n_sample = n_sample
        self.momentum = momentum
        self.class_per_num = self.n_sample // self.low_dim * ratio
        self.count = 0
        self.count_2 = 0

        self.classes_index = torch.tensor(list(range(self.low_dim))).cuda()
        self.register_buffer('classes', (torch.rand(self.n_sample) * self.low_dim).long())
        self.register_buffer('class_num', torch.zeros(self.low_dim).long())
        self.register_buffer('memory', torch.rand(self.n_sample, self.low_dim))
        pass

    def update_label(self, out, indexes):
        updated_weight = self.memory.index_select(0, indexes.data.view(-1)).resize_as_(out)
        updated_weight.mul_(self.momentum).add_(torch.mul(out.data, 1 - self.momentum))
        updated_weight.div(updated_weight.pow(2).sum(1, keepdim=True).pow(0.5))
        top_k = updated_weight.topk(self.low_dim, dim=1)[1]

        top_k = top_k.cpu()
        batch_size = out.size(0)
        class_num = self.class_num.cpu()
        new_class_num = np.zeros(shape=(batch_size, self.low_dim), dtype=np.int)
        class_labels = np.zeros(shape=(batch_size,), dtype=np.int)
        classes_cpu = self.classes.cpu()
        indexes_cpu = indexes.cpu()
        for i in range(batch_size):
            for j_index, j in enumerate(top_k[i]):
                if self.class_per_num > class_num[j]:
                    class_labels[i] += j
                    new_class_num[i][j] += 1
                    if classes_cpu[indexes_cpu[i]] != j:
                        self.count += 1
                    if j_index != 0:
                        self.count_2 += 1
                    break
                pass
            pass
        new_class_num = torch.tensor(new_class_num).long().cuda()
        self.class_num.index_copy_(0, self.classes_index, self.class_num + new_class_num.sum(0))

        # update
        class_labels = torch.tensor(class_labels).long().cuda()
        self.classes.index_copy_(0, indexes, class_labels)
        self.memory.index_copy_(0, indexes, updated_weight)
        return class_labels

    def forward(self, out, indexes, is_update=False, is_reset=False):
        if is_update:
            if is_reset:
                self.count = 0
                self.count_2 = 0
                self.class_num.index_copy_(0, self.classes_index, torch.zeros(self.low_dim).long().cuda())
                pass
            classes = self.update_label(out, indexes)
        else:
            classes = self.classes.index_select(0, indexes.data.view(-1)).resize_as_(indexes)
        return classes

    pass


class AttentionRunner(object):

    def __init__(self, low_dim=512, low_dim2=128, low_dim3=10, momentum=0.5,
                 learning_rate=0.03, learning_rate_type=0, linear_bias=True,
                 max_epoch=1000, t_epoch=300, first_epoch=200, resume=False,
                 checkpoint_path="./ckpt.t7", pre_train=None, data_root='./data'):
        self.learning_rate = learning_rate
        self.checkpoint_path = Tools.new_dir(checkpoint_path)
        self.resume = resume
        self.pre_train = pre_train
        self.data_root = data_root
        self.momentum = momentum
        self.low_dim = low_dim
        self.low_dim2 = low_dim2
        self.low_dim3 = low_dim3

        self.t_epoch = t_epoch
        self.max_epoch = max_epoch
        self.first_epoch = first_epoch
        self.linear_bias = linear_bias

        self.best_acc = 0

        self.adjust_learning_rate = self._adjust_learning_rate \
            if learning_rate_type == 0 else self._adjust_learning_rate2

        self.train_set, self.train_loader, self.test_set, self.test_loader, self.class_name = CIFAR10Instance.data(
            self.data_root)
        self.train_num = self.train_set.__len__()

        self.net = AttentionResNet(AttentionBasicBlock, [2, 2, 2, 2],
                                   self.low_dim, self.low_dim2, self.low_dim3, linear_bias=linear_bias).cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

        self._load_model(self.net)

        self.produce_class = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim, ratio=3).cuda()
        self.produce_class2 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim2, ratio=2).cuda()
        self.produce_class3 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim3, ratio=1).cuda()
        self.criterion = AttentionLoss().cuda()  # define loss function
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        pass

    def _adjust_learning_rate(self, epoch):

        def _get_lr(_base_lr, now_epoch, _t_epoch=self.t_epoch, _eta_min=1e-05):
            return _eta_min + (_base_lr - _eta_min) * (1 + math.cos(math.pi * now_epoch / _t_epoch)) / 2

        t_epoch = self.t_epoch
        first_epoch = self.first_epoch
        if epoch < first_epoch + self.t_epoch * 0:  # 0-300
            learning_rate = self.learning_rate
        elif epoch < first_epoch + t_epoch * 1:  # 300-400
            learning_rate = _get_lr(self.learning_rate / 1.0, epoch - first_epoch - t_epoch * 0)
        elif epoch < first_epoch + t_epoch * 2:  # 400-500
            learning_rate = _get_lr(self.learning_rate / 1.0, epoch - first_epoch - t_epoch * 1)
        elif epoch < first_epoch + t_epoch * 3:  # 500-600
            learning_rate = _get_lr(self.learning_rate / 3.0, epoch - first_epoch - t_epoch * 2)
        elif epoch < first_epoch + t_epoch * 4:  # 600-700
            learning_rate = _get_lr(self.learning_rate / 3.0, epoch - first_epoch - t_epoch * 3)
        elif epoch < first_epoch + t_epoch * 5:  # 700-800
            learning_rate = _get_lr(self.learning_rate / 10., epoch - first_epoch - t_epoch * 4)
        elif epoch < first_epoch + t_epoch * 6:  # 800-900
            learning_rate = _get_lr(self.learning_rate / 10., epoch - first_epoch - t_epoch * 5)
        else:  # 900-1000
            learning_rate = _get_lr(self.learning_rate / 30., epoch - first_epoch - t_epoch * 6)
            pass

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
            pass

        return learning_rate

    def _adjust_learning_rate2(self, epoch):
        if epoch < 100:
            learning_rate = self.learning_rate
        elif epoch < 200:
            learning_rate = self.learning_rate * 0.3
        elif epoch < 300:
            learning_rate = self.learning_rate * 0.1
        elif epoch < 400:
            learning_rate = self.learning_rate * 0.03
        else:
            learning_rate = self.learning_rate * 0.01
            pass

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
            pass
        return learning_rate

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

    def test(self, epoch=0, t=0.1, recompute_memory=1, loader_n=1):
        _acc = KNN.knn(epoch, self.net, self.produce_class, self.produce_class2,
                       self.produce_class3, self.train_loader, self.test_loader,
                       200, t, recompute_memory, loader_n=loader_n)
        return _acc

    def _train_one_epoch(self, epoch, update_epoch=3):

        # Update
        try:
            if epoch % update_epoch == 0:
                self.net.eval()
                Tools.print("Update label {} .......".format(epoch))
                for batch_idx, (inputs, _, indexes) in enumerate(self.train_loader):
                    inputs, indexes = inputs.cuda(), indexes.cuda()
                    out_logits, out_l2norm, out_logits2, out_l2norm2, out_logits3, out_l2norm3 = self.net(inputs)
                    self.produce_class(out_l2norm, indexes, True, True if batch_idx == 0 else False)
                    self.produce_class2(out_l2norm2, indexes, True, True if batch_idx == 0 else False)
                    self.produce_class3(out_l2norm3, indexes, True, True if batch_idx == 0 else False)
                    pass
                data_len = self.train_loader.dataset.__len__()
                Tools.print("Epoch: [{}] {}".format(epoch, [int(_) for _ in self.produce_class.class_num]))
                Tools.print("Epoch: [{}] {}".format(epoch, [int(_) for _ in self.produce_class2.class_num]))
                Tools.print("Epoch: [{}] {}".format(epoch, [int(_) for _ in self.produce_class3.class_num]))
                Tools.print("Epoch: [{}] 1 {}/{}-{}".format(epoch, self.produce_class.count,
                                                            self.produce_class.count_2, data_len))
                Tools.print("Epoch: [{}] 2 {}/{}-{}".format(epoch, self.produce_class2.count,
                                                            self.produce_class2.count_2, data_len))
                Tools.print("Epoch: [{}] 3 {}/{}-{}".format(epoch, self.produce_class3.count,
                                                            self.produce_class3.count_2, data_len))

                pass
        finally:
            pass

        # Train
        try:
            self.net.train()
            _learning_rate_ = self.adjust_learning_rate(epoch)
            Tools.print('Epoch: {} {}'.format(epoch, _learning_rate_))

            avg_loss_1 = AverageMeter()
            avg_loss_1_1 = AverageMeter()
            avg_loss_1_2 = AverageMeter()

            avg_loss_2 = AverageMeter()
            avg_loss_2_1 = AverageMeter()
            avg_loss_2_2 = AverageMeter()

            avg_loss_3 = AverageMeter()
            avg_loss_3_1 = AverageMeter()
            avg_loss_3_2 = AverageMeter()

            for batch_idx, (inputs, _, indexes) in enumerate(self.train_loader):
                inputs, indexes = inputs.cuda(), indexes.cuda()
                self.optimizer.zero_grad()

                out_logits, out_l2norm, out_logits2, out_l2norm2, out_logits3, out_l2norm3 = self.net(inputs)
                targets = self.produce_class(out_l2norm, indexes)
                targets2 = self.produce_class2(out_l2norm2, indexes)
                targets3 = self.produce_class3(out_l2norm3, indexes)

                params = [_ for _ in self.net.module.parameters()]
                loss_1, loss_1_1, loss_1_2 = self.criterion(out_logits, targets, params[-3])
                loss_2, loss_2_1, loss_2_2 = self.criterion(out_logits2, targets2, params[-2])
                loss_3, loss_3_1, loss_3_2 = self.criterion(out_logits3, targets3, params[-1])

                avg_loss_1.update(loss_1.item(), inputs.size(0))
                avg_loss_1_1.update(loss_1_1.item(), inputs.size(0))
                avg_loss_1_2.update(loss_1_2.item(), inputs.size(0))

                avg_loss_2.update(loss_2.item(), inputs.size(0))
                avg_loss_2_1.update(loss_2_1.item(), inputs.size(0))
                avg_loss_2_2.update(loss_2_2.item(), inputs.size(0))

                avg_loss_3.update(loss_3.item(), inputs.size(0))
                avg_loss_3_1.update(loss_3_1.item(), inputs.size(0))
                avg_loss_3_2.update(loss_3_2.item(), inputs.size(0))

                if batch_idx % 3 == 0:
                    loss_1.backward()
                elif batch_idx % 3 == 1:
                    loss_2.backward()
                else:
                    loss_3.backward()
                    pass

                self.optimizer.step()
                pass

            Tools.print(
                'Epoch: {}/{} Loss 1: {:.4f}({:.4f}/{:.4f}) '
                'Loss 2: {:.4f}({:.4f}/{:.4f}) Loss 3: {:.4f}({:.4f}/{:.4f})'.format(
                    epoch, len(self.train_loader), avg_loss_1.avg, avg_loss_1_1.avg, avg_loss_1_2.avg, avg_loss_2.avg,
                    avg_loss_2_1.avg, avg_loss_2_2.avg, avg_loss_3.avg, avg_loss_3_1.avg, avg_loss_3_2.avg))
        finally:
            pass

        # Test
        try:
            Tools.print("Test {} .......".format(epoch))
            _acc = self.test(epoch=epoch, recompute_memory=0)
            if _acc > self.best_acc:
                Tools.print('Saving..')
                state = {'net': self.net.state_dict(), 'acc': _acc, 'epoch': epoch}
                torch.save(state, self.checkpoint_path)
                self.best_acc = _acc
                pass
            Tools.print('best accuracy: {:.2f}'.format(self.best_acc * 100))
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    """
    First train auto-encoder and then train my method.
    
    Top 1: 83.19(8192, 13310/4694), 82.57(512, 6783/1585), 82.11(64, 5593/1424)|(1000+500+500+500)
    """

    # _low_dim = 2048
    # _low_dim2 = 512
    # _low_dim3 = 128

    # _low_dim = 1024
    # _low_dim2 = 256
    # _low_dim3 = 64

    _low_dim = 8192
    _low_dim2 = 512
    _low_dim3 = 64

    _start_epoch = 0
    _max_epoch = 500
    _learning_rate = 0.001
    _learning_rate_type = 1
    _linear_bias = False

    _first_epoch = 300
    _t_epoch = 100

    _momentum = 0.5
    _resume = False

    # _pre_train = None
    _pre_train = "./checkpoint/11_class_8192_3level_512_64_500_1_l1/ckpt.t7"
    _name = "11_class_{}_3level_{}_{}_{}_{}_l1".format(_low_dim, _low_dim2, _low_dim3,
                                                      _max_epoch, 0 if _linear_bias else 1)
    _checkpoint_path = "./checkpoint/{}/ckpt.t7".format(_name)

    Tools.print()
    Tools.print("low_dim={} low_dim2={} low_dim3={} name={} pre_train={} momentum={} checkpoint_path={}".format(
        _low_dim, _low_dim2, _low_dim3, _name, _pre_train, _momentum, _checkpoint_path))
    Tools.print()

    runner = AttentionRunner(low_dim=_low_dim, low_dim2=_low_dim2, low_dim3=_low_dim3, momentum=_momentum,
                             linear_bias=_linear_bias, learning_rate=_learning_rate,
                             learning_rate_type=_learning_rate_type,
                             max_epoch=_max_epoch, t_epoch=_t_epoch, first_epoch=_first_epoch,
                             resume=_resume, pre_train=_pre_train, checkpoint_path=_checkpoint_path)

    Tools.print()
    acc = runner.test()
    Tools.print('Random accuracy: {:.2f}'.format(acc * 100))

    runner.train(start_epoch=_start_epoch, update_epoch=1)

    Tools.print()
    acc = runner.test(loader_n=2)
    Tools.print('final accuracy: {:.2f}'.format(acc * 100))
    pass
