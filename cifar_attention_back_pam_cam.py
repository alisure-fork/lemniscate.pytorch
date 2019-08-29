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
    def forward(self, cam_features, indexes, cam_memory, params):
        # v^T_i*v/t
        t = params[0].item()
        out = torch.mm(cam_features.data, cam_memory.t()).div_(t)

        self.save_for_backward(cam_features, indexes, cam_memory, params)
        return out

    @staticmethod
    def backward(self, grad_output):
        cam_features, indexes, cam_memory, params = self.saved_tensors

        # add temperature and gradient of linear
        t = params[0].item()
        grad_output.data.div_(t)
        grad_input = torch.mm(grad_output.data, cam_memory)
        grad_input.resize_as_(cam_features)
        return grad_input, None, None, None

    pass


class NearestAttention(nn.Module):

    def __init__(self, n_sample, k_nearest, pam_size, cam_size, t=0.1, momentum=0.5):
        super(NearestAttention, self).__init__()
        self.n_sample = n_sample
        self.k_nearest = k_nearest
        self.pam_size = pam_size
        self.cam_size = cam_size

        self.t = t
        self.momentum = momentum

        self.softmax = nn.Softmax(dim=-1)

        std_pam = 1. / math.sqrt(self.pam_size / 2)
        std_cam = 1. / math.sqrt(self.cam_size / 2)
        self.register_buffer('params', torch.tensor([self.t, self.momentum]))
        self.register_buffer('pam_memory', torch.rand(self.n_sample, self.pam_size).mul_(2 * std_pam).add_(-std_pam))
        self.register_buffer('cam_memory', torch.rand(self.n_sample, self.cam_size).mul_(2 * std_cam).add_(-std_cam))
        pass

    def forward(self, pam_features, cam_features, indexes):
        # k_nearest_indexes
        k_score, k_pam_features, k_cam_features = self._get_k_nearest(pam_features.data,
                                                                      cam_features.data, self.k_nearest)

        # update features
        self.update_features(pam_features, cam_features, indexes)

        # no parametric instance discrimination
        linear_average_out = LinearNPIDOp.apply(cam_features, indexes, self.cam_memory, self.params)

        return k_score, k_pam_features, k_cam_features, linear_average_out

    def update_features(self, pam_features, cam_features, indexes):
        # update pam
        weight_pos = self.pam_memory.index_select(0, indexes.data.view(-1)).resize_as_(pam_features)
        weight_pos.mul_(self.momentum)
        weight_pos.add_(torch.mul(pam_features.data, 1 - self.momentum))
        self.pam_memory.index_copy_(0, indexes, weight_pos)

        # update cam
        weight_pos = self.cam_memory.index_select(0, indexes.data.view(-1)).resize_as_(cam_features)
        weight_pos.mul_(self.momentum)
        weight_pos.add_(torch.mul(cam_features.data, 1 - self.momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        self.cam_memory.index_copy_(0, indexes, updated_weight)
        pass

    def init_features(self, pam_features, cam_features, indexes):
        # init pam
        self.pam_memory.index_copy_(0, indexes, pam_features.data)
        # init cam
        self.cam_memory.index_copy_(0, indexes, cam_features.data)
        pass

    def _get_k_nearest(self, pam_features, cam_features, k_nearest):
        dist = torch.mm(cam_features, self.cam_memory.t())
        k_nearest_score, k_nearest_indexes = dist.topk(k_nearest, dim=1, largest=True, sorted=True)
        k_nearest_score = self.softmax(k_nearest_score)

        # k_nearest_pam_features
        k_nearest_pam_features = self.pam_memory.index_select(
            0, k_nearest_indexes.data.view(-1)).resize_(pam_features.size(0), self.k_nearest, self.pam_size)

        # k_nearest_cam_features
        k_nearest_cam_features = self.cam_memory.index_select(
            0, k_nearest_indexes.data.view(-1)).resize_(cam_features.size(0), self.k_nearest, self.cam_size)

        return k_nearest_score, k_nearest_pam_features, k_nearest_cam_features

    pass


class KNN(object):

    @staticmethod
    def knn(epoch, net, nearest_attention, train_loader, test_loader, k, t, recompute_memory=0):
        net.eval()

        top1 = 0.
        top5 = 0.
        total = 0
        train_features = nearest_attention.cam_memory.t()
        train_labels = torch.LongTensor(train_loader.dataset.train_labels).cuda()
        c = train_labels.max() + 1
        test_size = test_loader.dataset.__len__()

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

        with torch.no_grad():
            retrieval_one_hot = torch.zeros(k, c).cuda()  # [200, 10]
            for batch_idx, (inputs, targets, indexes) in enumerate(test_loader):
                targets = targets.cuda(async=True)
                feature_final, _ = net(inputs)

                dist = torch.mm(feature_final, train_features)

                # ---------------------------------------------------------------------------------- #
                batch_size = inputs.size(0)
                yd, yi = dist.topk(k, dim=1, largest=True, sorted=True)
                candidates = train_labels.view(1, -1).expand(batch_size, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval_one_hot.resize_(batch_size * k, c).zero_()
                retrieval_one_hot = retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1).view(batch_size, -1, c)
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
                        epoch, total, test_size, top1 * 100. / total, top5 * 100. / total))
                pass
            pass

        Tools.print("Test {} Top1={:.2f} Top5={:.2f}".format(epoch, top1 * 100. / total, top5 * 100. / total))
        return top1 / total

    pass


class PAMModule(nn.Module):

    def __init__(self, pam_size):
        super(PAMModule, self).__init__()
        self.pam_size = pam_size
        self.softmax = nn.Softmax(dim=-1)
        pass

    def forward(self, k_nearest_score, pam_features, k_nearest_pam_features):
        pam_features = pam_features.data
        k_nearest_pam_features = k_nearest_pam_features.data

        batch_size, k_nearest = k_nearest_score.size()
        height, width, channel = self.pam_size

        proj_query = pam_features.view(batch_size, channel, height*width).permute(0, 2, 1)
        proj_key = k_nearest_pam_features.view(batch_size, k_nearest, channel, width*height).permute(1, 0, 2, 3)

        out_k = []
        for i in range(k_nearest):
            energy = torch.bmm(proj_query, proj_key[i])
            attention = self.softmax(energy)
            out = torch.bmm(proj_key[i], attention.permute(0, 2, 1))
            out = out.view(batch_size, 1, -1)
            out_k.append(out)
            pass
        cat_out = torch.cat(out_k, dim=1)
        out = torch.bmm(k_nearest_score.view(batch_size, 1, -1), cat_out).squeeze()
        return out

    pass


class CAMModule(nn.Module):

    def __init__(self, cam_size):
        self.cam_size = cam_size
        super(CAMModule, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        pass

    def forward(self, k_nearest_score, cam_features, k_nearest_cam_features):
        k_nearest_cam_features = k_nearest_cam_features.data

        batch_size, k_nearest = k_nearest_score.size()
        out = torch.bmm(k_nearest_score.view(batch_size, 1, -1), k_nearest_cam_features).squeeze()
        return out

    pass


class AttentionLoss(nn.Module):

    def __init__(self):
        super(AttentionLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.criterion_no = nn.CrossEntropyLoss()
        pass

    def forward(self, pam_features, pam_features_2, cam_features, cam_features_2, linear_average_out, indexes):
        loss_1 = self.criterion(pam_features, pam_features_2) * 20
        loss_2 = self.criterion(cam_features, cam_features_2) * 10
        loss_3 = self.criterion_no(linear_average_out, indexes)
        return loss_3, loss_1, loss_2, loss_3

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

        self.net, self.nearest_attention, self.pam_module, self.cam_module, self.criterion = self._build_model()

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

        nearest_attention = NearestAttention(self.train_num, self.k_nearest,
                                             net.module.pam_size_one, net.module.cam_size_one, self.momentum)
        pam_module = PAMModule(net.module.pam_size)
        cam_module = CAMModule(net.module.cam_size)

        criterion = AttentionLoss()  # define loss function

        if self.pre_train:
            Tools.print('==> Pre train from checkpoint {} ..'.format(self.pre_train))
            checkpoint = torch.load(self.pre_train)
            net.load_state_dict(checkpoint['net'])
            if "nearest_attention" in checkpoint.keys():
                Tools.print('==> Pre train load nearest_attention')
                nearest_attention = checkpoint['nearest_attention']
            pass

        # Load checkpoint.
        if self.resume:
            Tools.print('==> Resuming from checkpoint {} ..'.format(self.checkpoint_path))
            checkpoint = torch.load(self.checkpoint_path)
            net.load_state_dict(checkpoint['net'])
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
            if "nearest_attention" in checkpoint.keys():
                Tools.print('==> Resuming load nearest_attention')
                nearest_attention = checkpoint['nearest_attention']
            Tools.print("{} {}".format(self.best_acc, self.start_epoch))
            pass

        net = net.to(self.device)
        nearest_attention = nearest_attention.to(self.device)
        pam_module = pam_module.to(self.device)
        cam_module = cam_module.to(self.device)
        criterion = criterion.to(self.device)

        return net, nearest_attention, pam_module, cam_module, criterion

    def init_memory(self):
        self.net.eval()

        transform_bak = self.train_loader.dataset.transform
        self.train_loader.dataset.transform = self.test_loader.dataset.transform
        temp_loader = torch.utils.data.DataLoader(self.train_loader.dataset,
                                                  batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, _, indexes) in enumerate(temp_loader):
            inputs, indexes = inputs.to(self.device), indexes.to(self.device)
            _, features = self.net(inputs)
            self.nearest_attention.init_features(features["layer3"], features["final"], indexes)
            if batch_idx % 100 == 0:
                Tools.print("Init {}/{}".format(batch_idx, len(temp_loader)))
            pass
        self.train_loader.dataset.transform = transform_bak
        pass

    def test(self, epoch=0, t=0.1, recompute_memory=1):
        _acc = KNN.knn(epoch, self.net, self.nearest_attention,
                       self.train_loader, self.test_loader, 200, t, recompute_memory)
        return _acc

    def _train_one_epoch(self, epoch, max_epoch):
        Tools.print()
        Tools.print('Epoch: %d' % epoch)

        self.net.train()
        self._adjust_learning_rate(epoch, max_epoch)

        avg_loss = AverageMeter()
        avg_loss_1 = AverageMeter()
        avg_loss_2 = AverageMeter()
        avg_loss_3 = AverageMeter()
        for batch_idx, (inputs, _, indexes) in enumerate(self.train_loader):
            inputs, indexes = inputs.to(self.device), indexes.to(self.device)
            self.optimizer.zero_grad()

            feature_final, features = self.net(inputs)
            pam_features = features["layer3"]
            cam_features = features["final"]

            k_nearest_score, k_pam_features, k_cam_features, linear_average_out = self.nearest_attention(
                pam_features, cam_features, indexes)
            pam_features_2 = self.pam_module(k_nearest_score, pam_features, k_pam_features)
            cam_features_2 = self.cam_module(k_nearest_score, cam_features, k_cam_features)

            loss, loss_1, loss_2, loss_3 = self.criterion(pam_features, pam_features_2,
                                                          cam_features, cam_features_2,
                                                          linear_average_out, indexes)
            avg_loss.update(loss.item(), inputs.size(0))
            avg_loss_1.update(loss_1.item(), inputs.size(0))
            avg_loss_2.update(loss_2.item(), inputs.size(0))
            avg_loss_3.update(loss_3.item(), inputs.size(0))

            loss.backward()

            self.optimizer.step()

            if batch_idx % 50 == 0:
                Tools.print(
                    'Epoch: [{}][{}/{}] '
                    'Loss +: {avg_loss.val:.4f} ({avg_loss.avg:.4f}) '
                    'Loss 1: {avg_loss_1.val:.4f} ({avg_loss_1.avg:.4f}) '
                    'Loss 2: {avg_loss_2.val:.4f} ({avg_loss_2.avg:.4f}) '
                    'Loss 3: {avg_loss_3.val:.4f} ({avg_loss_3.avg:.4f})'.format(
                        epoch, batch_idx, len(self.train_loader), avg_loss=avg_loss,
                        avg_loss_1=avg_loss_1, avg_loss_2=avg_loss_2, avg_loss_3=avg_loss_3))
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
                    # 'nearest_attention': self.nearest_attention,
                }
                torch.save(state, self.checkpoint_path)
                self.best_acc = _acc
                pass

            Tools.print('best accuracy: {:.2f}'.format(self.best_acc * 100))
        pass

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    _k_nearest = 25

    pre_train = None
    # pre_train = "./checkpoint/attention_resume/ckpt.t7"
    runner = AttentionRunner(k_nearest=_k_nearest, resume=False, pre_train=pre_train,
                             checkpoint_path="./checkpoint/attention_test_2/ckpt.t7")

    Tools.print()
    acc = runner.test()
    Tools.print('random accuracy: {:.2f}'.format(acc * 100))

    # init memory
    Tools.print()
    Tools.print("Init memory")
    runner.init_memory()

    Tools.print()
    acc = runner.test(recompute_memory=0)
    Tools.print('Init accuracy: {:.2f}'.format(acc * 100))

    runner.train(epoch_num=300)

    Tools.print()
    acc = runner.test()
    Tools.print('final accuracy: {:.2f}'.format(acc * 100))
    pass
