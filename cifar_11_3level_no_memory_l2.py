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
    def knn(epoch, net, low_dim, low_dim2, low_dim3, train_loader, test_loader, k, t, loader_n=1):
        net.eval()
        n_sample = train_loader.dataset.__len__()
        out_memory = torch.rand(n_sample, low_dim).t().cuda()
        out_memory2 = torch.rand(n_sample, low_dim2).t().cuda()
        out_memory3 = torch.rand(n_sample, low_dim3).t().cuda()
        train_labels = torch.LongTensor(train_loader.dataset.train_labels).cuda()
        max_c = train_labels.max() + 1

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
                top1, top5 = 0., 0.
                top12, top52 = 0., 0.
                top13, top53 = 0., 0.
                total = 0

                retrieval_one_hot = torch.zeros(k, max_c).cuda()  # [200, 10]
                retrieval_one_hot2 = torch.zeros(k, max_c).cuda()  # [200, 10]
                retrieval_one_hot3 = torch.zeros(k, max_c).cuda()  # [200, 10]
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


class AttentionRunner(object):

    def __init__(self, low_dim=512, low_dim2=128, low_dim3=10,
                 learning_rate=0.03, learning_rate_type=0, linear_bias=True, has_l1=False,
                 max_epoch=1000, t_epoch=300, first_epoch=200, resume=False,
                 checkpoint_path="./ckpt.t7", pre_train=None, data_root='./data'):
        self.learning_rate = learning_rate
        self.checkpoint_path = Tools.new_dir(checkpoint_path)
        self.resume = resume
        self.pre_train = pre_train
        self.data_root = data_root

        self.low_dim = low_dim
        self.low_dim2 = low_dim2
        self.low_dim3 = low_dim3

        self.t_epoch = t_epoch
        self.max_epoch = max_epoch
        self.first_epoch = first_epoch
        self.linear_bias = linear_bias
        self.has_l1 = has_l1
        self.learning_rate_type = learning_rate_type
        self.adjust_learning_rate = self._adjust_learning_rate \
            if self.learning_rate_type == 0 else self._adjust_learning_rate2

        self.best_acc = 0

        self.train_set, self.train_loader, self.test_set, self.test_loader, self.class_name = CIFAR10Instance.data(
            self.data_root)
        self.train_num = self.train_set.__len__()

        self.net = AttentionResNet(AttentionBasicBlock, [2, 2, 2, 2],
                                   self.low_dim, self.low_dim2, self.low_dim3, linear_bias=linear_bias).cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

        self._load_model(self.net)

        self.produce_class = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim, ratio=3)
        self.produce_class2 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim2, ratio=2)
        self.produce_class3 = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim3, ratio=1)
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
            learning_rate = self.learning_rate * 0.5
        elif epoch < 300:
            learning_rate = self.learning_rate * 0.1
        elif epoch < 400:
            learning_rate = self.learning_rate * 0.05
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

    def test(self, epoch=0, t=0.1, loader_n=1):
        _acc = KNN.knn(epoch, self.net, self.low_dim, self.low_dim2, self.low_dim3,
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
                for batch_idx, (inputs, _, indexes) in enumerate(self.train_loader):
                    inputs, indexes = inputs.cuda(), indexes.cuda()
                    out_logits, out_l2norm, out_logits2, out_l2norm2, out_logits3, out_l2norm3 = self.net(inputs)
                    self.produce_class.cal_label(out_l2norm, indexes)
                    self.produce_class2.cal_label(out_l2norm2, indexes)
                    self.produce_class3.cal_label(out_l2norm3, indexes)
                    pass
                Tools.print("Epoch: [{}] 1-{}/{} 2-{}/{} 3-{}/{}".format(
                    epoch, self.produce_class2.count, self.produce_class2.count_2,
                    self.produce_class2.count, self.produce_class2.count_2,
                    self.produce_class3.count, self.produce_class3.count_2))
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

                targets = self.produce_class.get_label(indexes)
                targets2 = self.produce_class2.get_label(indexes)
                targets3 = self.produce_class3.get_label(indexes)

                params = [_ for _ in self.net.module.parameters()]
                loss_1, loss_1_1, loss_1_2 = self.criterion(out_logits, targets, params[-3])
                loss_2, loss_2_1, loss_2_2 = self.criterion(out_logits2, targets2, params[-2])
                loss_3, loss_3_1, loss_3_2 = self.criterion(out_logits3, targets3, params[-1])

                # Tools.print('Epoch: [{}/{}] {}-{}-{} / {}-{}-{} / {}-{}-{}'.format(
                #     epoch, len(self.train_loader),
                #     loss_1.item(), loss_1_1.item(), loss_1_2.item(),
                #     loss_2.item(), loss_2_1.item(), loss_2_2.item(),
                #     loss_3.item(), loss_3_1.item(), loss_3_2.item()))

                avg_loss_1.update(loss_1.item(), inputs.size(0))
                avg_loss_1_1.update(loss_1_1.item(), inputs.size(0))
                avg_loss_1_2.update(loss_1_2.item(), inputs.size(0))

                avg_loss_2.update(loss_2.item(), inputs.size(0))
                avg_loss_2_1.update(loss_2_1.item(), inputs.size(0))
                avg_loss_2_2.update(loss_2_2.item(), inputs.size(0))

                avg_loss_3.update(loss_3.item(), inputs.size(0))
                avg_loss_3_1.update(loss_3_1.item(), inputs.size(0))
                avg_loss_3_2.update(loss_3_2.item(), inputs.size(0))

                if self.has_l1:
                    if batch_idx % 3 == 0:
                        loss_1.backward()
                    elif batch_idx % 3 == 1:
                        loss_2.backward()
                    else:
                        loss_3.backward()
                else:
                    if batch_idx % 3 == 0:
                        loss_1_1.backward()
                    elif batch_idx % 3 == 1:
                        loss_2_1.backward()
                    else:
                        loss_3_1.backward()
                    pass

                self.optimizer.step()
                pass

            Tools.print(
                'Epoch: {}/{} Loss 1: {:.4f}({:.4f}/{:.4f}) '
                'Loss 2: {:.4f}({:.4f}/{:.4f}) Loss 3: {:.4f}({:.4f}/{:.4f})'.format(
                    epoch, len(self.train_loader), avg_loss_1.avg, avg_loss_1_1.avg, avg_loss_1_2.avg,avg_loss_2.avg,
                    avg_loss_2_1.avg, avg_loss_2_2.avg, avg_loss_3.avg, avg_loss_3_1.avg, avg_loss_3_2.avg))
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
            if self.learning_rate_type == 1 and self.max_epoch >= 1000 and epoch > self.max_epoch // 2:
                self.has_l1 = True
            self._train_one_epoch(epoch, update_epoch=update_epoch)
            pass
        pass

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    """
    First train auto-encoder and then train my method.
    
    Top 1: 84.78(1024, 23651/4686), 84.84(256, 19331/4524), 85.05(64, 14853/2513)|(1000+500+500)
    Top 1: 85.24(1024, 22236/4551), 85.36(256, 18167/4526), 85.46(64, 13803/2485)|(1000+500+500+500)
    """

    # _low_dim, _low_dim2, _low_dim3 = 2048, 512, 128
    # _low_dim, _low_dim2, _low_dim3 = 1024, 256, 64
    _low_dim, _low_dim2, _low_dim3 = 8192, 1024, 128

    _start_epoch = 0
    _max_epoch = 1001
    _learning_rate = 0.005
    _learning_rate_type = 1

    # _start_epoch = 300
    # _max_epoch = 1000
    # _learning_rate = 0.01
    # _learning_rate_type = 0

    _has_l1 = False
    _linear_bias = False

    _first_epoch, _t_epoch = 300, 100

    _resume = True

    _pre_train = None
    # _pre_train = "./checkpoint/11_class_1024_3level_256_64_500_no_memory_1_l1/ckpt.t7"
    _name = "11_class_{}_3level_{}_{}_{}_no_memory_{}_l1".format(_low_dim, _low_dim2, _low_dim3,
                                                                 _max_epoch, 0 if _linear_bias else 1)
    _checkpoint_path = "./checkpoint/{}/ckpt.t7".format(_name)

    Tools.print()
    Tools.print("low_dim={} low_dim2={} low_dim3={} name={} pre_train={} checkpoint_path={}".format(
        _low_dim, _low_dim2, _low_dim3, _name, _pre_train, _checkpoint_path))
    Tools.print()

    runner = AttentionRunner(low_dim=_low_dim, low_dim2=_low_dim2, low_dim3=_low_dim3,
                             linear_bias=_linear_bias, has_l1=_has_l1,
                             learning_rate=_learning_rate, learning_rate_type=_learning_rate_type,
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
