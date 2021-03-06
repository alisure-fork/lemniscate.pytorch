import os
import math
import torch
import models
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Function
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms


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

    pass


class LinearNPIDOp(Function):

    @staticmethod
    def forward(self, final_features, indexes, memory, params):
        # v^T_i*v/t
        t = params[0].item()
        out = torch.mm(final_features.data, memory.t()).div_(t)

        self.save_for_backward(final_features, indexes, memory, params)
        return out

    @staticmethod
    def backward(self, grad_output):
        final_features, indexes, memory, params = self.saved_tensors

        # add temperature and gradient of linear
        t = params[0].item()
        grad_output.data.div_(t)
        grad_input = torch.mm(grad_output.data, memory)
        grad_input.resize_as_(final_features)
        return grad_input, None, None, None

    pass


class NearestAttention(nn.Module):

    def __init__(self, n_sample, k_nearest, final_size, t=0.1, momentum=0.5):
        super(NearestAttention, self).__init__()
        self.n_sample = n_sample
        self.k_nearest = k_nearest
        self.final_size = final_size

        self.t = t
        self.momentum = momentum

        self.softmax = nn.Softmax(dim=-1)

        std_final = 1. / math.sqrt(self.final_size / 2)
        self.register_buffer('params', torch.tensor([self.t, self.momentum]))
        self.register_buffer('memory', torch.rand(self.n_sample, self.final_size).mul_(2 * std_final).add_(-std_final))
        pass

    def forward(self, final_features, indexes):
        # k_nearest_indexes
        k_invert_pam_features = self._get_k_nearest(final_features.data, self.k_nearest)

        # update features
        self.update_features(final_features, indexes)

        # no parametric instance discrimination
        linear_average_out = LinearNPIDOp.apply(final_features, indexes, self.memory, self.params)

        return k_invert_pam_features, linear_average_out

    def update_features(self, final_features, indexes):
        # update final
        weight_pos = self.memory.index_select(0, indexes.data.view(-1)).resize_as_(final_features)
        weight_pos.mul_(self.momentum)
        weight_pos.add_(torch.mul(final_features.data, 1 - self.momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        self.memory.index_copy_(0, indexes, updated_weight)
        pass

    def init_features(self, final_features, indexes):
        # init final memory
        self.memory.index_copy_(0, indexes, final_features.data)
        pass

    def _get_k_nearest(self, final_features, k_nearest):
        # dist
        dist = torch.mm(final_features, self.memory.t())

        # K invert
        _, k_invert_indexes = dist.topk(k_nearest, dim=1, largest=False, sorted=True)
        k_invert_features = self.memory.index_select(
            0, k_invert_indexes.data.view(-1)).resize_(final_features.size(0), self.k_nearest, self.final_size)

        return k_invert_features

    pass


class KNN(object):

    @staticmethod
    def knn(epoch, net, nearest_attention, train_loader, test_loader, k, t,
            recompute_memory=0, loader_n=1):
        net.eval()

        train_features = nearest_attention.memory.t()
        train_labels = torch.LongTensor(train_loader.dataset.train_labels).cuda()
        c = train_labels.max() + 1

        if recompute_memory:
            transform_bak = train_loader.dataset.transform

            train_loader.dataset.transform = test_loader.dataset.transform
            temp_loader = torch.utils.data.DataLoader(train_loader.dataset,
                                                      batch_size=100, shuffle=False, num_workers=1)
            for batch_idx, (inputs, _, indexes) in enumerate(temp_loader):
                feature_final, _ = net(inputs)
                batch_size = inputs.size(0)
                train_features[:, batch_idx * batch_size:batch_idx * batch_size + batch_size] = feature_final.data.t()
                pass

            train_loader.dataset.transform = transform_bak
            pass

        accs = []
        with torch.no_grad():
            retrieval_one_hot = torch.zeros(k, c).cuda()  # [200, 10]
            now_loader = [test_loader] if loader_n == 1 else [test_loader, train_loader]
            for loader in now_loader:
                top1 = 0.
                top5 = 0.
                total = 0

                smaple_number = loader.dataset.__len__()
                for batch_idx, (inputs, targets, indexes) in enumerate(loader):
                    targets = targets.cuda(async=True)
                    feature_final, _ = net(inputs)

                    dist = torch.mm(feature_final, train_features)

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

                    if batch_idx % 50 == 0:
                        Tools.print('Test {} [{}/{}] Top1: {:.2f}  Top5: {:.2f}'.format(
                            epoch, total, smaple_number, top1 * 100. / total, top5 * 100. / total))
                    pass

                Tools.print("Test {} Top1={:.2f} Top5={:.2f}".format(epoch, top1 * 100. / total, top5 * 100. / total))
                accs.append(top1 / total)
                pass
            pass

        return accs[0]

    pass


class AttentionLoss(nn.Module):

    def __init__(self, k_nearest):
        super(AttentionLoss, self).__init__()
        self.k_nearest = k_nearest
        self.criterion_no = nn.CrossEntropyLoss()
        self.criterion = nn.CosineSimilarity()
        pass

    def forward(self, final_features, final_features_invert, linear_average_out, indexes):
        loss_1 = self.criterion_no(linear_average_out, indexes)
        loss_2 = (1 - self.criterion(final_features, final_features_invert)).sum()
        # return loss_1 + loss_2, loss_1, loss_2
        return loss_2, loss_1, loss_2

    pass


class AttentionRunner(object):

    def __init__(self, k_nearest=11, momentum=0.5, learning_rate=0.03, resume=False,
                 checkpoint_path="./ckpt.t7", pre_train=None, data_root='./data'):
        self.learning_rate = learning_rate
        self.checkpoint_path = Tools.new_dir(checkpoint_path)
        self.resume = resume
        self.pre_train = pre_train
        self.data_root = data_root
        self.k_nearest = k_nearest
        self.momentum = momentum

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_acc = 0
        self.start_epoch = 0

        self.train_set, self.train_loader, self.test_set, self.test_loader, self.class_name = self._data()
        self.train_num = self.train_set.__len__()

        self.net, self.nearest_attention, self.criterion = self._build_model()

        self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        pass

    def _data(self):
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

        train_set = CIFAR10Instance(root=self.data_root, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers=2)

        test_set = CIFAR10Instance(root=self.data_root, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

        class_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return train_set, train_loader, test_set, test_loader, class_name

    def _adjust_learning_rate(self, epoch, max_epoch=200):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        learning_rate = self.learning_rate * (0.1 ** (
                (epoch - max_epoch // 3) // (max_epoch // 6))) if epoch >= max_epoch // 3 else self.learning_rate
        Tools.print("learning rate={}".format(learning_rate))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
            pass
        pass

    def _build_model(self):
        Tools.print('==> Building model..')
        net = models.__dict__['AttentionResNet18'](low_dim=128)

        if self.device == 'cuda':
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True
            pass

        nearest_attention = NearestAttention(self.train_num, self.k_nearest, net.module.cam_size_one, self.momentum)

        criterion = AttentionLoss(self.k_nearest)  # define loss function

        if self.pre_train:
            Tools.print('==> Pre train from checkpoint {} ..'.format(self.pre_train))
            checkpoint = torch.load(self.pre_train)
            net.load_state_dict(checkpoint['net'])
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

        net = net.to(self.device)
        nearest_attention = nearest_attention.to(self.device)
        criterion = criterion.to(self.device)

        return net, nearest_attention, criterion

    def init_memory(self):
        self.net.eval()

        transform_bak = self.train_loader.dataset.transform
        self.train_loader.dataset.transform = self.test_loader.dataset.transform
        temp_loader = torch.utils.data.DataLoader(self.train_loader.dataset,
                                                  batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, _, indexes) in enumerate(temp_loader):
            inputs, indexes = inputs.to(self.device), indexes.to(self.device)
            _, features = self.net(inputs)
            self.nearest_attention.init_features(features["final"], indexes)
            if batch_idx % 100 == 0:
                Tools.print("Init {}/{}".format(batch_idx, len(temp_loader)))
            pass
        self.train_loader.dataset.transform = transform_bak
        pass

    def test(self, epoch=0, t=0.1, recompute_memory=1, loader_n=1):
        _acc = KNN.knn(epoch, self.net, self.nearest_attention,
                       self.train_loader, self.test_loader, 200, t, recompute_memory, loader_n=loader_n)
        return _acc

    def _train_one_epoch(self, epoch, max_epoch):
        Tools.print()
        Tools.print('Epoch: %d' % epoch)

        self.net.train()
        self._adjust_learning_rate(epoch, max_epoch)

        avg_loss = AverageMeter()
        avg_loss_1 = AverageMeter()
        avg_loss_2 = AverageMeter()
        for batch_idx, (inputs, _, indexes) in enumerate(self.train_loader):
            inputs, indexes = inputs.to(self.device), indexes.to(self.device)
            self.optimizer.zero_grad()

            feature_final, features = self.net(inputs)

            feature_final_invert, linear_average_out = self.nearest_attention(feature_final, indexes)
            feature_final_invert = -feature_final_invert.mean(dim=1)
            feature_final_invert = feature_final_invert.div(feature_final_invert.pow(2).sum(1, keepdim=True).pow(0.5))

            loss, loss_1, loss_2 = self.criterion(feature_final, feature_final_invert, linear_average_out, indexes)
            avg_loss.update(loss.item(), inputs.size(0))
            avg_loss_1.update(loss_1.item(), inputs.size(0))
            avg_loss_2.update(loss_2.item(), inputs.size(0))

            loss.backward()

            self.optimizer.step()

            if batch_idx % 50 == 0:
                Tools.print(
                    'Epoch: [{}][{}/{}] '
                    'Loss +: {avg_loss.val:.4f} ({avg_loss.avg:.4f}) '
                    'Loss 1: {avg_loss_1.val:.4f} ({avg_loss_1.avg:.4f}) '
                    'Loss 2: {avg_loss_2.val:.4f} ({avg_loss_2.avg:.4f})'.format(
                        epoch, batch_idx, len(self.train_loader), avg_loss=avg_loss,
                        avg_loss_1=avg_loss_1, avg_loss_2=avg_loss_2))
                pass

            pass

        pass

    def train(self, epoch_num=200):
        for epoch in range(self.start_epoch, self.start_epoch + epoch_num):
            self._train_one_epoch(epoch, max_epoch=self.start_epoch + epoch_num)
            _acc = self.test(epoch=epoch, recompute_memory=0)

            if _acc > self.best_acc:
                Tools.print('Saving..')
                state = {
                    'net': self.net.state_dict(),
                    'acc': _acc,
                    'epoch': epoch,
                }
                torch.save(state, self.checkpoint_path)
                self.best_acc = _acc
                pass

            Tools.print('best accuracy: {:.2f}'.format(self.best_acc * 100))
        pass

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    """
    Top 1: 
    """

    _k_nearest = 25

    pre_train = None
    # pre_train = "./checkpoint/attention_resume/ckpt.t7"
    runner = AttentionRunner(k_nearest=_k_nearest, resume=False, pre_train=pre_train,
                             checkpoint_path="./checkpoint/attention_single_invert/ckpt.t7")

    # Tools.print()
    # acc = runner.test()
    # Tools.print('Random accuracy: {:.2f}'.format(acc * 100))
    #
    # init memory
    Tools.print()
    Tools.print("Init memory")
    runner.init_memory()
    #
    # Tools.print()
    # acc = runner.test(recompute_memory=0)
    # Tools.print('Init accuracy: {:.2f}'.format(acc * 100))

    runner.train(epoch_num=300)

    Tools.print()
    acc = runner.test(loader_n=2)
    Tools.print('final accuracy: {:.2f}'.format(acc * 100))
    pass
