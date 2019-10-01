import os
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

    def __init__(self, block, num_blocks, low_dim=128):
        super(AttentionResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear_256 = nn.Linear(512 * block.expansion, low_dim)
        self.softmax = nn.Softmax(dim=-1)
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
        out_logits = self.linear_256(out)
        out_softmax = self.softmax(out_logits)
        return out_logits, out_softmax

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
    def knn(epoch, net, produce_class, train_loader, test_loader, k, t, recompute_memory=0, loader_n=1):
        net.eval()

        out_memory = produce_class.memory.t()
        train_labels = torch.LongTensor(train_loader.dataset.train_labels).cuda()
        c = train_labels.max() + 1

        if recompute_memory:
            transform_bak = train_loader.dataset.transform

            train_loader.dataset.transform = test_loader.dataset.transform
            temp_loader = torch.utils.data.DataLoader(train_loader.dataset, 100, shuffle=False, num_workers=1)
            for batch_idx, (inputs, _, indexes) in enumerate(temp_loader):
                out_logits, out = net(inputs)
                batch_size = inputs.size(0)
                out_memory[:, batch_idx * batch_size:batch_idx * batch_size + batch_size] = out.data.t()
                pass

            train_loader.dataset.transform = transform_bak
            pass

        all_acc = []
        with torch.no_grad():
            now_loader = [test_loader] if loader_n == 1 else [test_loader, train_loader]

            for loader in now_loader:
                top1 = 0.
                top5 = 0.
                total = 0

                sample_number = loader.dataset.__len__()
                retrieval_one_hot = torch.zeros(k, c).cuda()  # [200, 10]
                for batch_idx, (inputs, targets, indexes) in enumerate(loader):
                    targets = targets.cuda(async=True)
                    out_logits, out = net(inputs)
                    dist = torch.mm(out, out_memory)

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

                    total += targets.size(0)

                    if batch_idx % 100 == 0:
                        Tools.print('Test {} [{}/{}] Top1: {:.2f}  Top5: {:.2f}'.format(
                            epoch, total, sample_number, top1 * 100. / total, top5 * 100. / total))
                    pass

                Tools.print("Test {} Top1={:.2f} Top5={:.2f}".format(epoch, top1 * 100. / total, top5 * 100. / total))
                all_acc.append(top1 / total)

                pass
            pass

        return all_acc[0]

    pass


class AttentionLoss(nn.Module):

    def __init__(self):
        super(AttentionLoss, self).__init__()
        self.criterion_no = nn.CrossEntropyLoss()
        pass

    def forward(self, out, targets):
        loss_1 = self.criterion_no(out, targets)
        return loss_1

    pass


class ProduceClass(nn.Module):

    def __init__(self, n_sample, low_dim, momentum=0.5, candidate=8):
        super(ProduceClass, self).__init__()
        self.low_dim = low_dim
        self.n_sample = n_sample
        self.momentum = momentum
        self.candidate = candidate
        self.class_per_num = self.n_sample // self.low_dim
        self.classes_index = torch.tensor(list(range(self.low_dim))).cuda()

        self.register_buffer('classes', (torch.rand(self.n_sample) * self.low_dim).long())
        self.register_buffer('classes_candidate', (torch.rand(self.n_sample, self.candidate) * self.low_dim).long())
        self.register_buffer('class_num', torch.zeros(self.low_dim).long())
        self.register_buffer('memory', torch.rand(self.n_sample, self.low_dim))
        pass

    def update_label(self, out, indexes, is_first):
        features = self.memory.index_select(0, indexes.data.view(-1)).resize_as_(out)
        features.mul_(self.momentum).add_(torch.mul(out.data, 1 - self.momentum))
        top_k = features.topk(self.low_dim, dim=1)[1]

        top_k = top_k.cpu()
        batch_size = out.size(0)
        class_num = self.class_num.cpu()
        new_class_num = np.zeros(shape=(batch_size, self.low_dim), dtype=np.int)
        class_labels = np.zeros(shape=(batch_size,), dtype=np.int)
        _class_labels_candidate = self.classes_candidate[indexes].cpu()
        _class_labels_old = self.classes[indexes].cpu()

        count = 0
        for i in range(batch_size):
            if is_first:
                for j in top_k[i]:
                    if self.class_per_num > class_num[j]:
                        class_labels[i] = j
                        new_class_num[i][class_labels[i]] += 1
                        count += 1
                        break
                    pass
                pass
            else:
                _j = _class_labels_old[i]
                for j in top_k[i][: self.candidate]:
                    if j in _class_labels_candidate[i] and self.class_per_num > class_num[j]:
                        _j = j
                        count += 1
                        break
                    pass
                class_labels[i] = _j
                new_class_num[i][class_labels[i]] += 1
            pass
        new_class_num = torch.tensor(new_class_num).long().cuda()
        class_labels_candidate = top_k[:, :self.candidate].cuda()
        class_labels = torch.tensor(class_labels).long().cuda()

        # update
        self.classes.index_copy_(0, indexes, class_labels)
        self.classes_candidate.index_copy_(0, indexes, class_labels_candidate)
        self.class_num.index_copy_(0, self.classes_index, self.class_num + new_class_num.sum(0))
        self.memory.index_copy_(0, indexes, features)
        return class_labels, count

    def forward(self, out, indexes, is_update=False, is_reset=False, is_first=False):
        if is_update:
            if is_reset:
                self.class_num.index_copy_(0, self.classes_index, torch.zeros(self.low_dim).long().cuda())
            return self.update_label(out, indexes, is_first)
        else:
            return self.classes.index_select(0, indexes.data.view(-1)).resize_as_(indexes)
        pass

    pass


class AttentionRunner(object):

    def __init__(self, low_dim=128, momentum=0.5, learning_rate=0.03, resume=False,
                 checkpoint_path="./ckpt.t7", pre_train=None, data_root='./data'):
        self.learning_rate = learning_rate
        self.checkpoint_path = Tools.new_dir(checkpoint_path)
        self.resume = resume
        self.pre_train = pre_train
        self.data_root = data_root
        self.momentum = momentum
        self.low_dim = low_dim

        self.best_acc = 0
        self.start_epoch = 0

        self.train_set, self.train_loader, self.test_set, self.test_loader, self.class_name = CIFAR10Instance.data(
            self.data_root)
        self.train_num = self.train_set.__len__()

        self.net = AttentionResNet(AttentionBasicBlock, [2, 2, 2, 2], self.low_dim).cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

        self._load_model(self.net)

        self.produce_class = ProduceClass(n_sample=self.train_num, low_dim=self.low_dim).cuda()
        self.criterion = AttentionLoss().cuda()  # define loss function
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        pass

    def _adjust_learning_rate(self, epoch, max_epoch=200):
        learning_rate = self.learning_rate * (0.1 ** (
                (epoch - max_epoch // 3) // (max_epoch // 6))) if epoch >= max_epoch // 3 else self.learning_rate
        Tools.print("learning rate={}".format(learning_rate))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
            pass
        pass

    def _load_model(self, net):
        if self.pre_train:
            Tools.print('==> Pre train from checkpoint {} ..'.format(self.pre_train))
            checkpoint = torch.load(self.pre_train)
            net.load_state_dict(checkpoint['net'], strict=False)
            pass

        # Load checkpoint.
        if self.resume:
            Tools.print('==> Resuming from checkpoint {} ..'.format(self.checkpoint_path))
            checkpoint = torch.load(self.checkpoint_path)
            net.load_state_dict(checkpoint['net'])
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
            Tools.print("{} {}".format(self.best_acc, self.start_epoch))
            pass
        pass

    def test(self, epoch=0, t=0.1, recompute_memory=1, loader_n=1):
        _acc = KNN.knn(epoch, self.net, self.produce_class, self.train_loader, self.test_loader,
                       200, t, recompute_memory, loader_n=loader_n)
        return _acc

    def _train_one_epoch(self, epoch, max_epoch, update_epoch=3):

        # Update and Test
        if epoch % update_epoch == 0:
            self.net.eval()
            Tools.print()
            Tools.print("Update label {} .......".format(epoch))
            count_all = 0
            for batch_idx, (inputs, _, indexes) in enumerate(self.train_loader):
                inputs, indexes = inputs.cuda(), indexes.cuda()
                out_logits, out = self.net(inputs)
                classes, count = self.produce_class(out, indexes, True,
                                                    True if batch_idx == 0 else False,
                                                    True if epoch == 0 else False)
                count_all += count
                if batch_idx % 100 == 0:
                    Tools.print('Epoch: [{}][{}/{}] update={} - {})'.format(epoch, batch_idx, len(self.train_loader),
                                                                            count_all, [int(_) for _ in classes]))
                    pass
                pass

            Tools.print("Epoch: [{}] update={} {}".format(epoch, count_all,
                                                          [int(_) for _ in self.produce_class.class_num]))

            Tools.print()
            Tools.print("Test {} .......".format(epoch))
            _acc = self.test(epoch=epoch, recompute_memory=0)
            if _acc > self.best_acc:
                Tools.print()
                Tools.print('Saving..')
                state = {'net': self.net.state_dict(), 'acc': _acc, 'epoch': epoch}
                torch.save(state, self.checkpoint_path)
                self.best_acc = _acc
                pass
            Tools.print('best accuracy: {:.2f}'.format(self.best_acc * 100))
            pass

        # Train
        Tools.print()
        Tools.print('Epoch: %d' % epoch)
        self.net.train()
        self._adjust_learning_rate(epoch, max_epoch)
        avg_loss = AverageMeter()
        for batch_idx, (inputs, _, indexes) in enumerate(self.train_loader):
            inputs, indexes = inputs.cuda(), indexes.cuda()
            self.optimizer.zero_grad()

            out_logits, out = self.net(inputs)
            targets = self.produce_class(out, indexes)

            loss = self.criterion(out_logits, targets)
            avg_loss.update(loss.item(), inputs.size(0))
            loss.backward()
            self.optimizer.step()

            if batch_idx % 100 == 0:
                Tools.print('Epoch: [{}][{}/{}] Loss +: {avg_loss.val:.4f} ({avg_loss.avg:.4f})'.format(
                        epoch, batch_idx, len(self.train_loader), avg_loss=avg_loss))
                pass

            pass

        pass

    def train(self, epoch_num=200, update_epoch=3):
        for epoch in range(self.start_epoch, self.start_epoch + epoch_num):
            self._train_one_epoch(epoch, max_epoch=self.start_epoch + epoch_num, update_epoch=update_epoch)
            pass
        pass

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    """
    Top 1: 
    """

    _low_dim = 256
    _name = "5_class_{}_softmax_candidate".format(_low_dim)

    _momentum = 0.5
    _pre_train = None
    # _pre_train = "./checkpoint/{}/ckpt.t7".format(_name)
    _checkpoint_path = "./checkpoint/{}/ckpt.t7".format(_name)

    Tools.print()
    Tools.print("low_dim={} name={} pre_train={} momentum={} checkpoint_path={}".format(
        _low_dim, _name, _pre_train, _momentum, _checkpoint_path))
    Tools.print()

    runner = AttentionRunner(low_dim=_low_dim, learning_rate=0.03, momentum=_momentum, resume=False,
                             pre_train=_pre_train, checkpoint_path=_checkpoint_path)

    Tools.print()
    acc = runner.test()
    Tools.print('Random accuracy: {:.2f}'.format(acc * 100))

    runner.train(epoch_num=300, update_epoch=1)

    Tools.print()
    acc = runner.test(loader_n=2)
    Tools.print('final accuracy: {:.2f}'.format(acc * 100))
    pass
