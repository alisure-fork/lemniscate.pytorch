import os
import torch
import numpy as np
import torchvision
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from alisuretool.Tools import Tools
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as torch_utils_data


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


class AutoEncoderOld(nn.Module):

    def __init__(self, low_dim=512):
        super(AutoEncoderOld, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),  # [batch, 64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # [batch, 128, 8, 8]
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # [batch, 256, 4, 4]
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # [batch, 512, 2, 2]
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2)  # [batch, 512, 1, 1]
        )

        self.linear = nn.Linear(512, low_dim)
        self.l2norm = Normalize(2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(low_dim, 512, 4, stride=2, padding=1),  # [batch, 512, 2, 2]
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # [batch, 256, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # [batch, 128, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # [batch, 64, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )
        pass

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        linear = self.linear(encoded)
        l2norm = self.l2norm(linear)
        decoded = self.decoder(linear.view(linear.size(0), -1, 1, 1))
        return encoded, linear, l2norm, decoded

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


class AutoEncoder(nn.Module):

    def __init__(self, block, num_blocks, low_dim=128):
        super(AutoEncoder, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512 * block.expansion, low_dim)
        self.l2norm = Normalize(2)

        self.decoder = self._decoder(low_dim)
        pass

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        encoded = out.view(out.size(0), -1)

        linear = self.linear(encoded)
        l2norm = self.l2norm(linear)
        decoded = self.decoder(linear.view(linear.size(0), -1, 1, 1))
        return encoded, linear, l2norm, decoded

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    @staticmethod
    def _decoder(low_dim):
        decoder = nn.Sequential(
            nn.ConvTranspose2d(low_dim, 512, 4, stride=2, padding=1),  # [batch, 512, 2, 2]
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # [batch, 256, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # [batch, 128, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # [batch, 64, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )
        return decoder

    pass


class Data(object):

    @staticmethod
    def data(data_root, batch_size):
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        train_loader = torch_utils_data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        test_loader = torch_utils_data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return train_loader, test_loader, classes

    pass


class KNN(object):

    @staticmethod
    def knn(epoch, net, train_loader, test_loader, k, t=0.1, low_dim=128, loader_n=1):
        net.eval()

        sample_number = train_loader.dataset.__len__()
        out_memory = torch.rand(sample_number, low_dim).cuda().t()
        train_labels = torch.LongTensor(train_loader.dataset.train_labels).cuda()
        c = train_labels.max() + 1

        transform_bak = train_loader.dataset.transform
        train_loader.dataset.transform = test_loader.dataset.transform
        temp_loader = torch.utils.data.DataLoader(train_loader.dataset, 100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, _) in enumerate(temp_loader):
            inputs = inputs.cuda()
            encoded, linear, l2norm, decoded = net(inputs)
            batch_size = inputs.size(0)
            out_memory[:, batch_idx * batch_size:batch_idx * batch_size + batch_size] = l2norm.data.t()
            pass
        train_loader.dataset.transform = transform_bak

        all_acc = []
        with torch.no_grad():
            now_loader = [test_loader] if loader_n == 1 else [test_loader, train_loader]

            for loader in now_loader:
                top1, top5, total = 0., 0., 0
                retrieval_one_hot = torch.zeros(k, c).cuda()  # [200, 10]
                for batch_idx, (inputs, targets) in enumerate(loader):
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    encoded, linear, l2norm, decoded = net(inputs)
                    dist = torch.mm(l2norm, out_memory)

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

                    correct = predictions.eq(targets.data.view(-1, 1))

                    top1 += correct.narrow(1, 0, 1).sum().item()
                    top5 += correct.narrow(1, 0, 5).sum().item()

                    total += targets.size(0)
                    pass

                Tools.print("Test {} Top1={:.2f} Top5={:.2f}".format(epoch, top1 * 100. / total, top5 * 100. / total))
                all_acc.append(top1 / total)

                pass
            pass

        return all_acc[0]

    pass


class Runner(object):

    def __init__(self, data_root='./data', batch_size=32, low_dim=512,
                 checkpoint_path="./checkpoint/7_auto_encoder/ckpt.t7"):
        self.checkpoint_path = Tools.new_dir(checkpoint_path)

        self.low_dim = low_dim
        self.auto_encoder = AutoEncoder(AttentionBasicBlock, [2, 2, 2, 2], low_dim=self.low_dim).cuda()

        self.train_loader, self.test_loader, self.classes = Data.data(data_root, batch_size=batch_size)

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.auto_encoder.parameters())
        pass

    def train(self, max_epoch=100):
        for epoch in range(max_epoch):
            running_loss = 0.0
            for i, (inputs, _) in enumerate(self.train_loader):
                inputs = Variable(inputs.cuda())

                encoded, linear, l2norm, decoded = self.auto_encoder(inputs)
                loss = self.criterion(decoded, inputs)
                running_loss += loss.data

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pass
            Tools.print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(self.train_loader)))

            Tools.print('Saving Model...')
            torch.save(self.auto_encoder.state_dict(), self.checkpoint_path)
            if epoch % 10 == 0:
                self.test(epoch)
            pass
        Tools.print('Finished Training')

        Tools.print('Saving Model...')
        torch.save(self.auto_encoder.state_dict(), self.checkpoint_path)
        pass

    def inference(self):
        Tools.print("Loading checkpoint...")
        self.auto_encoder.load_state_dict(torch.load(self.checkpoint_path))

        test_iter = iter(self.test_loader)
        images, labels = next(test_iter)

        print('GroundTruth: ', ' '.join('%5s' % self.classes[labels[j]] for j in range(16)))
        show_data_1 = np.asarray(np.transpose(torchvision.utils.make_grid(images).cpu().numpy(),
                                              (1, 2, 0)) * 255, np.uint8)
        shape = show_data_1.shape

        images = Variable(images.cuda())
        encoded, linear, l2norm, decoded_img = self.auto_encoder(images)
        show_data_2 = np.asarray(np.transpose(torchvision.utils.make_grid(decoded_img.data).cpu().numpy(),
                                              (1, 2, 0)) * 255, np.uint8)

        padding = 5
        show = np.zeros(shape=(shape[0] + padding * 2, shape[1] * 2 + padding * 3, shape[2]), dtype=np.uint8)
        show[padding:-padding, padding:padding + shape[1], :] = show_data_1
        show[padding:-padding, padding * 2 + shape[1]:-padding, :] = show_data_2
        Image.fromarray(show).show()
        pass

    def test(self, epoch):
        return KNN.knn(epoch, self.auto_encoder, self.train_loader, self.test_loader, 200, low_dim=self.low_dim)

    def print_model(self):
        Tools.print()
        Tools.print("============== Encoder ==============")
        print(self.auto_encoder.encoder)
        Tools.print("============== Decoder ==============")
        print(self.auto_encoder.decoder)
        Tools.print()
        pass

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    _data_root = './data'
    _batch_size = 64
    _max_epoch = 100
    _checkpoint_path = "./checkpoint/7_auto_encoder_liner2/ckpt.t7"

    runner = Runner(data_root=_data_root, batch_size=_batch_size, checkpoint_path=_checkpoint_path)
    # runner.print_model()
    runner.test(0)
    runner.train(max_epoch=_max_epoch)
    runner.inference()
    pass
