import os
import sys
import time
import torch
import shutil
import pickle
import numpy as np
import torch.optim
from PIL import Image
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.nn.functional as F
from alisuretool.Tools import Tools
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
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

    def __init__(self, block, num_blocks, num_classes=1000, linear_bias=True):
        super(HCResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=linear_bias)
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
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out_logits = self.fc(out)
        return out_logits

    pass


class AverageMeter(object):

    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

        self.reset()
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

    def __str__(self):
        s = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return s.format(**self.__dict__)

    pass


class ProgressMeter(object):

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        pass

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        Tools.print(' '.join(entries))
        pass

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    pass


class DownSampledImageNetInstance(datasets.VisionDataset):
    base_folder = 'downsampled-imagenet'
    train_list = ['train_data_batch_1', 'train_data_batch_2', 'train_data_batch_3', 'train_data_batch_4',
                  'train_data_batch_5', 'train_data_batch_6', 'train_data_batch_7', 'train_data_batch_8',
                  'train_data_batch_9', 'train_data_batch_10', ]
    test_list = ['val_data']

    def __init__(self, root, image_size=32, train=True, transform=None, target_transform=None):
        super(DownSampledImageNetInstance, self).__init__(root, transform=transform, target_transform=target_transform)

        self.image_size = image_size
        self.train = train  # training set or test set
        downloaded_list = self.train_list if self.train else self.test_list

        self.data = []
        self.targets = []
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, str(self.image_size), file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
                pass
            pass

        self.targets = [v - 1 for v in self.targets] if np.max(self.targets) >= 1000 else self.targets
        self.data = np.vstack(self.data).reshape(-1, 3, self.image_size, self.image_size)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        pass

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

    @staticmethod
    def data(data_root, batch_size=128, image_size=32):
        Tools.print('==> Preparing data..')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])

        # transform_train = transforms.Compose([transforms.RandomResizedCrop(image_size),
        #                                       transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])

        transform_test = transforms.Compose([transforms.Resize(image_size),
                                             transforms.CenterCrop(image_size), transforms.ToTensor(), normalize])

        train_set = DownSampledImageNetInstance(root=data_root, image_size=image_size,
                                                train=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        test_set = DownSampledImageNetInstance(root=data_root, image_size=image_size,
                                               train=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

        class_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return train_set, train_loader, test_set, test_loader, class_name

    pass


class RunnerSingle(object):

    def __init__(self, lr=0.1, print_freq=10, start_epoch=0, epochs=90, image_size=32, num_classes=1000,
                 batch_size=256, workers=30, momentum=0.9, weight_decay=1e-4,
                 data_root="/home/z840/ALISURE/Data/tiny-imagenet-200",
                 resume_filename="./checkpoint/tiny_imagenet/ResNet18/checkpoint.pth.tar",
                 checkpoint_filename="./checkpoint/tiny_imagenet/ResNet18/checkpoint.pth.tar",
                 best_checkpoint_filename="./checkpoint/tiny_imagenet/ResNet18/checkpoint_best.pth.tar"):
        self.resume_filename = resume_filename
        self.checkpoint_filename = Tools.new_dir(checkpoint_filename)
        self.best_checkpoint_filename = best_checkpoint_filename

        self.lr = lr
        self.image_size = image_size
        self.num_classes = num_classes
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.print_freq = print_freq
        self.has_pretrained = False
        self.data_root = data_root
        self.batch_size = batch_size
        self.workers = workers
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.best_acc1 = 0

        (self.train_set, self.train_loader, self.test_set, self.test_loader, self.class_name
         ) = DownSampledImageNetInstance.data(self.data_root, batch_size=self.batch_size, image_size=self.image_size)

        self.model = HCResNet(HCBasicBlock, [2, 2, 2, 2], num_classes=self.num_classes, linear_bias=False).cuda()
        self.model = torch.nn.DataParallel(self.model)

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr,
                                         momentum=self.momentum, weight_decay=self.weight_decay)

        self._resume()
        pass

    def _resume(self):
        if os.path.isfile(self.resume_filename):
            checkpoint = torch.load(self.resume_filename)
            self.start_epoch = checkpoint['epoch']
            self.best_acc1 = checkpoint['best_acc1'].cuda()  # best_acc1
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            Tools.print("=> loaded checkpoint '{}' (epoch={}, acc={})".format(
                self.resume_filename, self.start_epoch, self.best_acc1))
        else:
            Tools.print("=> no checkpoint found at '{}'".format(self.resume_filename))

        cudnn.benchmark = True
        pass

    def _save_checkpoint(self, state, is_best):
        torch.save(state, self.checkpoint_filename)
        if is_best:
            shutil.copyfile(self.checkpoint_filename, self.best_checkpoint_filename)
        pass

    def _adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr * (0.1 ** (epoch // 10))

        # lr = 0.1
        # if epoch < 40:
        #     lr = 0.01
        # elif epoch < 50:
        #     lr = 0.001

        Tools.print("epoch={} lr={}".format(epoch, lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        pass

    @staticmethod
    def _accuracy(output, target, top_k=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            max_k = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(max_k, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
        pass

    def _train_one_epoch(self, epoch):
        batch_time, data_time = AverageMeter('Time', ':6.3f'), AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1, top5 = AverageMeter('Acc@1', ':6.2f'), AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(self.train_loader),
                                 [batch_time, data_time, losses, top1, top5], prefix="Epoch:[{}]".format(epoch))
        self.model.train()

        def _each_iter(images, target):
            images = images.cuda()
            target = target.cuda()

            output = self.model(images)
            loss = self.criterion(output, target)

            acc1, acc5 = self._accuracy(output, target, top_k=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pass

        end = time.time()
        for i, (_images, _target, _) in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            end = time.time()

            _each_iter(_images, _target)

            batch_time.update(time.time() - end)
            end = time.time()
            if i % self.print_freq == 0:
                progress.display(i)
                pass
            pass

        pass

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self._adjust_learning_rate(epoch)

            self._train_one_epoch(epoch)

            acc1 = self.validate()
            self.best_acc1 = max(acc1, self.best_acc1)

            self._save_checkpoint({'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                                   'best_acc1': self.best_acc1, 'optimizer': self.optimizer.state_dict()},
                                  acc1 >= self.best_acc1)
            pass
        pass

    def validate(self):
        batch_time, losses = AverageMeter('Time', ':6.3f'), AverageMeter('Loss', ':.4e')
        top1, top5 = AverageMeter('Acc@1', ':6.2f'), AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(self.test_loader), [batch_time, losses, top1, top5], prefix='Test: ')

        self.model.eval()
        with torch.no_grad():

            end = time.time()
            for i, (images, target, _) in enumerate(self.test_loader):
                images = images.cuda()
                target = target.cuda()

                output = self.model(images)
                loss = self.criterion(output, target)

                acc1, acc5 = self._accuracy(output, target, top_k=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.print_freq == 0:
                    progress.display(i)
                    pass
                pass

            Tools.print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
            pass

        return top1.avg

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 0 28.066 (32 expansion=1 strip=2,2,2,2,2)
    # 1 40.186 (32 expansion=1 strip=1,2,2,2,2)
    # 2 47.864 (32 expansion=1 strip=1,1,2,2,2)
    # 3 52.408 (32 expansion=1 strip=1,1,1,2,2)

    # 0 48.340 (64 expansion=1 strip=2,2,2,2,2)
    # 1 54.900 (64 expansion=1 strip=1,2,2,2,2)
    # 2 60.936 (64 expansion=1 strip=1,1,2,2,2) Acc@5 83.660
    # 3 (64 expansion=1 strip=1,1,1,2,2)

    _data_root = "./data"
    _arg = ["resnet18_64_2_data", 0.1, 256, 16, 64]
    RunnerSingle(lr=_arg[1], print_freq=200, start_epoch=0, epochs=30,
                 batch_size=_arg[2], workers=_arg[3], momentum=0.9, image_size=_arg[4], num_classes=1000,
                 weight_decay=1e-4, data_root=_data_root,
                 resume_filename="./checkpoint/downsampled_imagenet/{}/checkpoint.pth.tar".format(_arg[0]),
                 checkpoint_filename="./checkpoint/downsampled_imagenet/{}/checkpoint.pth.tar".format(_arg[0]),
                 best_checkpoint_filename="./checkpoint/downsampled_imagenet/{}/checkpoint_best.pth.tar".format(_arg[0])
                 ).train()
    pass
