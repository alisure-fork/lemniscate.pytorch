import math
import torch
import models
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from alisuretool.Tools import Tools
from torch.autograd import Function
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


class LinearAverageOp(Function):

    @staticmethod
    def forward(self, x, y, memory, params):
        # v^T_i*v/t
        out = torch.mm(x.data, memory.t())
        t = params[0].item()
        out.div_(t)

        self.save_for_backward(x, memory, y, params)
        return out

    @staticmethod
    def backward(self, grad_output):
        x, memory, y, params = self.saved_tensors

        # add temperature and gradient of linear
        t = params[0].item()
        grad_output.data.div_(t)
        grad_input = torch.mm(grad_output.data, memory)
        grad_input.resize_as_(x)

        # update the non-parametric data
        momentum = params[1].item()
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return grad_input, None, None, None

    pass


class LinearAverage(nn.Module):

    def __init__(self, input_size, n, t=0.07, momentum=0.5):
        super(LinearAverage, self).__init__()
        self.n = n
        self.t = t
        self.momentum = momentum

        std_v = 1. / math.sqrt(input_size / 3)
        self.register_buffer('params', torch.tensor([self.t, self.momentum]))
        self.register_buffer('memory', torch.rand(self.n, input_size).mul_(2 * std_v).add_(-std_v))
        pass

    def forward(self, x, y):
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out

    pass


class NCEAverageOp(Function):

    @staticmethod
    def forward(self, x, y, memory, idx, params):
        m = int(params[0].item())
        t = params[1].item()
        z = params[2].item()

        batch_size = x.size(0)
        output_size = memory.size(0)
        input_size = memory.size(1)

        # sample positives & negatives
        idx.select(1, 0).copy_(y.data)
        # sample correspoinding weights
        weight = torch.index_select(memory, 0, idx.view(-1))
        weight.resize_(batch_size, m + 1, input_size)

        # inner product
        out = torch.bmm(weight, x.data.resize_(batch_size, input_size, 1))
        out.div_(t).exp_()  # batch_size * self.k+1
        x.data.resize_(batch_size, input_size)

        if z < 0:
            params[2] = out.mean() * output_size
            z = params[2].item()
            Tools.print("normalization constant Z is set to {:.1f}".format(z))
            pass

        out.div_(z).resize_(batch_size, m+ 1)

        self.save_for_backward(x, memory, y, weight, out, params)
        return out

    @staticmethod
    def backward(self, grad_output):
        x, memory, y, weight, out, params = self.saved_tensors

        batch_size = grad_output.size(0)
        m = int(params[0].item())
        t = params[1].item()
        momentum = params[3].item()

        # add temperature and gradient of linear
        grad_output.data.mul_(out.data)
        grad_output.data.div_(t)
        grad_output.data.resize_(batch_size, 1, m + 1)
        grad_input = torch.bmm(grad_output.data, weight)
        grad_input.resize_as_(x)

        # update the non-parametric data
        weight_pos = weight.select(1, 0).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return grad_input, None, None, None, None

    pass


class NCEAverage(nn.Module):

    def __init__(self, input_size, n, m, t=0.07, momentum=0.5, z=-1):
        super(NCEAverage, self).__init__()
        self.n = n
        self.m = m
        self.z = z
        self.t = t
        self.momentum = momentum

        std_v = 1. / math.sqrt(input_size / 3)
        self.register_buffer('params', torch.tensor([self.m, self.t, self.z, self.momentum]))
        self.register_buffer('memory', torch.rand(self.n, input_size).mul_(2 * std_v).add_(-std_v))
        pass

    def forward(self, x, y):
        batch_size = x.size(0)
        idx = torch.zeros(batch_size * (self.m + 1), dtype=torch.long).cuda().random_(0, self.n).view(batch_size, -1)
        out = NCEAverageOp.apply(x, y, self.memory, idx, self.params)
        return out

    pass


class KNN(object):

    @staticmethod
    def nn(net, nce_avg, train_loader, test_loader, recompute_memory=0):
        net.eval()

        correct = 0.
        total = 0

        train_features = nce_avg.memory.t()
        train_labels = torch.LongTensor(train_loader.dataset.train_labels).cuda()
        test_size = test_loader.dataset.__len__()

        if recompute_memory:
            transform_bak = train_loader.dataset.transform

            train_loader.dataset.transform = test_loader.dataset.transform
            temp_loader = torch.utils.data.DataLoader(train_loader.dataset,
                                                      batch_size=100, shuffle=False, num_workers=1)
            for batch_idx, (inputs, _, indexes) in enumerate(temp_loader):
                features = net(inputs)
                batch_size = inputs.size(0)
                train_features[:, batch_idx * batch_size:batch_idx * batch_size + batch_size] = features.data.t()
                pass

            train_loader.dataset.transform = transform_bak
            pass

        with torch.no_grad():
            for batch_idx, (inputs, targets, indexes) in enumerate(test_loader):
                targets = targets.cuda(async=True)
                features = net(inputs)

                dist = torch.mm(features, train_features)
                yd, yi = dist.topk(1, dim=1, largest=True, sorted=True)

                candidates = train_labels.view(1, -1).expand(inputs.size(0), -1)
                retrieval = torch.gather(candidates, 1, yi)
                retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)
                correct += retrieval.eq(targets.data).sum().item()

                total += targets.size(0)
                Tools.print('Test [{}/{}] Top1: {:.2f}'.format(total, test_size, correct * 100. / total))

        return correct / total

    @staticmethod
    def knn(epoch, net, nce_avg, train_loader, test_loader, k, sigma, recompute_memory=0):
        net.eval()

        top1 = 0.
        top5 = 0.
        total = 0

        train_features = nce_avg.memory.t()
        train_labels = torch.LongTensor(train_loader.dataset.train_labels).cuda()
        c = train_labels.max() + 1
        test_size = test_loader.dataset.__len__()

        if recompute_memory:
            transform_bak = train_loader.dataset.transform

            train_loader.dataset.transform = test_loader.dataset.transform
            temp_loader = torch.utils.data.DataLoader(train_loader.dataset,
                                                      batch_size=100, shuffle=False, num_workers=1)
            for batch_idx, (inputs, _, indexes) in enumerate(temp_loader):
                features = net(inputs)
                batch_size = inputs.size(0)
                train_features[:, batch_idx * batch_size:batch_idx * batch_size + batch_size] = features.data.t()
                pass

            train_loader.dataset.transform = transform_bak
            pass

        with torch.no_grad():
            retrieval_one_hot = torch.zeros(k, c).cuda()  # [200, 10]
            for batch_idx, (inputs, targets, indexes) in enumerate(test_loader):
                targets = targets.cuda(async=True)
                features = net(inputs)

                dist = torch.mm(features, train_features)

                batch_size = inputs.size(0)
                yd, yi = dist.topk(k, dim=1, largest=True, sorted=True)
                candidates = train_labels.view(1, -1).expand(batch_size, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval_one_hot.resize_(batch_size * k, c).zero_()
                retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                yd_transform = yd.clone().div_(sigma).exp_()
                probs = torch.sum(torch.mul(retrieval_one_hot.view(batch_size, -1, c),
                                            yd_transform.view(batch_size, -1, 1)), 1)
                _, predictions = probs.sort(1, True)

                # Find which predictions match the target
                correct = predictions.eq(targets.data.view(-1, 1))

                top1 = top1 + correct.narrow(1, 0, 1).sum().item()
                top5 = top5 + correct.narrow(1, 0, 5).sum().item()

                total += targets.size(0)

                if batch_idx % 50 == 0:
                    Tools.print('Test {} [{}/{}] Top1: {:.2f}  Top5: {:.2f}'.format(
                        epoch, total, test_size, top1 * 100. / total, top5 * 100. / total))
                pass
            pass

        Tools.print("Test {} Top1={:.2f} Top5={:.2f}".format(epoch, top1 * 100. / total, top5 * 100. / total))
        return top1 / total

    pass


class NCECriterion(nn.Module):

    def __init__(self, train_num, eps=1e-7):
        super(NCECriterion, self).__init__()
        self.train_num = train_num
        self.eps = eps
        pass

    def forward(self, x, targets):
        m = x.size(1) - 1
        pn = 1 / float(self.train_num)

        # eq 5.1 : P(origin=model) = Pmt / (Pmt + m*Pnt)
        p_m = x.select(1, 0)
        p_m_add = p_m.add(m * pn + self.eps)
        p_model = torch.div(p_m, p_m_add)
        p_model.log_()
        p_model_sum = p_model.sum(0)

        # eq 5.2 : P(origin=noise) = m*Pns / (Pms + m*Pns)
        p_n_add = x.narrow(1, 1, m).add(m * pn + self.eps)
        p_n = p_n_add.clone().fill_(m * pn)
        p_noise = torch.div(p_n, p_n_add)
        p_noise.log_()
        p_noise_sum = p_noise.view(-1, 1).sum(0)

        loss = - (p_model_sum + p_noise_sum) / x.size(0)
        return loss

    pass


class NPIDRunner(object):

    def __init__(self, learning_rate=0.03, checkpoint_path="./checkpoint/ckpt.t7", resume=False,
                 data_root='./data', low_dim=128, nce_m=0, nce_t=0.1, nce_momentum=0.5):
        self.learning_rate = learning_rate
        self.checkpoint_path = Tools.new_dir(checkpoint_path)
        self.resume = resume
        self.data_root = data_root

        self.low_dim = low_dim
        self.nce_m = nce_m
        self.nce_t = nce_t
        self.nce_momentum = nce_momentum

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_acc = 0
        self.start_epoch = 0

        self.train_set, self.train_loader, self.test_set, self.test_loader, self.class_name = self._data()
        self.train_num = self.train_set.__len__()

        self.net, self.nce_avg, self.criterion = self._build_model()

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
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)

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
        net = models.__dict__['ResNet18'](low_dim=self.low_dim)

        if self.device == 'cuda':
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True
            pass

        # define loss function
        if self.nce_m > 0:
            nce_avg = NCEAverage(self.low_dim, self.train_num, self.nce_m, self.nce_t, self.nce_momentum)
            criterion = NCECriterion(self.train_num)
        else:
            nce_avg = LinearAverage(self.low_dim, self.train_num, self.nce_t, self.nce_momentum)
            criterion = nn.CrossEntropyLoss()
            pass

        # Load checkpoint.
        if self.resume:
            Tools.print('==> Resuming from checkpoint {} ..'.format(self.checkpoint_path))
            checkpoint = torch.load(self.checkpoint_path)
            net.load_state_dict(checkpoint['net'])
            nce_avg = checkpoint['nce_avg']
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
            Tools.print("{} {}".format(self.best_acc, self.start_epoch))
            pass

        return net.to(self.device), nce_avg.to(self.device), criterion.to(self.device)

    def test(self, epoch=0, recompute_memory=1):
        _acc = KNN.knn(epoch, self.net, self.nce_avg, self.train_loader,
                       self.test_loader, 200, self.nce_t, recompute_memory)
        return _acc

    def _train_one_epoch(self, epoch):
        Tools.print()
        Tools.print('Epoch: %d' % epoch)

        self.net.train()
        self._adjust_learning_rate(epoch)

        avg_loss = AverageMeter()
        for batch_idx, (inputs, _, indexes) in enumerate(self.train_loader):
            inputs, indexes = inputs.to(self.device), indexes.to(self.device)
            self.optimizer.zero_grad()

            features = self.net(inputs)
            outputs = self.nce_avg(features, indexes)
            loss = self.criterion(outputs, indexes)
            avg_loss.update(loss.item(), inputs.size(0))

            loss.backward()
            self.optimizer.step()

            if batch_idx % 50 == 0:
                Tools.print('Epoch: [{}][{}/{}] Loss: {avg_loss.val:.4f} ({avg_loss.avg:.4f})'.format(
                    epoch, batch_idx, len(self.train_loader), avg_loss=avg_loss))
                pass
            pass

        pass

    def train(self, epoch_num=200):
        for epoch in range(self.start_epoch, self.start_epoch + epoch_num):
            self._train_one_epoch(epoch)
            acc = self.test(epoch=epoch, recompute_memory=0)

            if acc > self.best_acc:
                Tools.print('Saving..')
                state = {
                    'net': self.net.state_dict(),
                    'nce_avg': self.nce_avg,
                    'acc': acc,
                    'epoch': epoch,
                }
                torch.save(state, self.checkpoint_path)
                self.best_acc = acc
                pass

            Tools.print('best accuracy: {:.2f}'.format(self.best_acc * 100))
        pass

    pass


if __name__ == '__main__':
    _nce_m = 1
    _resume = False
    runner = NPIDRunner(nce_m=_nce_m, checkpoint_path="./checkpoint/demo_nce_{}/ckpt.t7".format(_nce_m), resume=_resume)
    acc = runner.test()
    Tools.print('now accuracy: {:.2f}'.format(acc * 100))

    runner.train(epoch_num=200)

    acc = runner.test()
    Tools.print('final accuracy: {:.2f}'.format(acc * 100))
    pass
