import os
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optimizer
from alisuretool.Tools import Tools
import torchvision.datasets as data_set
import torchvision.transforms as transforms
from cifar_10_3level_no_memory_l2_sum import HCBasicBlock as AttentionBasicBlock


class CIFAR10Instance(data_set.CIFAR10):

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
    def data(data_root, is_train_shuffle=True):
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
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=is_train_shuffle, num_workers=2)

        test_set = CIFAR10Instance(root=data_root, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

        class_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return train_set, train_loader, test_set, test_loader, class_name

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


class LinearClassifier(nn.Module):

    def __init__(self, input_size, output_size=10):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        pass

    def forward(self, inputs):
        return self.linear(inputs)

    pass


class MultipleLinearClassifiers(nn.Module):

    def __init__(self, input_size_list, output_size=10):
        super(MultipleLinearClassifiers, self).__init__()
        self.linear_1 = LinearClassifier(input_size_list[0], input_size_list[1])
        self.linear_2 = LinearClassifier(input_size_list[1], input_size_list[2])
        self.linear_3 = LinearClassifier(input_size_list[2], output_size)
        pass

    def forward(self, inputs):
        return self.linear_3(self.linear_2(self.linear_1(inputs)))

    pass


class MultipleNonLinearClassifier(nn.Module):

    def __init__(self, input_size_list, output_size=10):
        super(MultipleNonLinearClassifier, self).__init__()

        self.linear_1 = nn.Linear(input_size_list[0], input_size_list[1], bias=False)
        self.bn_1 = nn.BatchNorm1d(input_size_list[1])
        self.relu_1 = nn.ReLU(inplace=True)

        self.linear_2 = nn.Linear(input_size_list[1], input_size_list[2], bias=False)
        self.bn_2 = nn.BatchNorm1d(input_size_list[2])
        self.relu_2 = nn.ReLU(inplace=True)

        self.linear_3 = nn.Linear(input_size_list[2], output_size)
        pass

    def forward(self, inputs):
        out = self.relu_1(self.bn_1(self.linear_1(inputs)))
        out = self.relu_2(self.bn_2(self.linear_2(out)))
        out = self.linear_3(out)
        return out

    pass


class Classifier(nn.Module):

    def __init__(self, low_dim, input_size_or_list, output_size=10,
                 classifier_type=0, which_out=0, is_fine_tune=False, linear_bias=True):
        super(Classifier, self).__init__()
        assert len(low_dim) * 2 > which_out

        self.which_out = which_out
        self.is_fine_tune = is_fine_tune

        self.attention = AttentionResNet(AttentionBasicBlock,
                                         [2, 2, 2, 2], *low_dim, linear_bias=linear_bias).cuda()

        if not self.is_fine_tune:
            for p in self.parameters():
                p.requires_grad = False
                pass
            pass

        if classifier_type == 1:
            assert isinstance(input_size_or_list, list) and len(input_size_or_list) >=1
            self.linear = MultipleLinearClassifiers(input_size_or_list, output_size)
        elif classifier_type == 2:
            assert isinstance(input_size_or_list, list) and len(input_size_or_list) >=1
            self.linear = MultipleNonLinearClassifier(input_size_or_list, output_size)
        else:
            assert isinstance(input_size_or_list, int)
            self.linear = LinearClassifier(input_size_or_list, output_size)
        pass

    def forward(self, inputs):
        out = self.attention(inputs)
        out = self.linear(out[self.which_out])
        return out
    pass


class ClassierRunner(object):

    def __init__(self, net, learning_rate=0.01, max_epoch=1000, resume=False, is_fine_tune=False,
                 pre_train_path=None, checkpoint_path="./classier.t7", data_root='./data'):
        self.learning_rate = learning_rate
        self.checkpoint_path = Tools.new_dir(checkpoint_path)
        self.pre_train_path = pre_train_path
        self.resume = resume
        self.is_fine_tune = is_fine_tune
        self.data_root = data_root

        self.best_acc = 0

        if self.is_fine_tune:
            self.learning_rate = 0.001
            pass

        self.train_set, self.train_loader, self.test_set, self.test_loader, _ = CIFAR10Instance.data(
            self.data_root, is_train_shuffle=True)
        self.train_num = self.train_set.__len__()

        self.net = net.cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

        self._load_model(self.net)

        self.criterion = nn.CrossEntropyLoss().cuda()  # define loss function
        self.optimizer = optimizer.SGD(filter(lambda p: p.requires_grad, self.net.parameters()),
                                       lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)

        self.max_epoch = max_epoch
        pass

    def _adjust_learning_rate(self, epoch):
        if epoch < 50:
            learning_rate = self.learning_rate
        elif epoch < 100:
            learning_rate = self.learning_rate * 0.1
        elif epoch < 150:
            learning_rate = self.learning_rate * 0.01
        else:
            learning_rate = self.learning_rate * 0.001
            pass

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
            pass
        return learning_rate

    def _load_model(self, net):
        # pre train
        if self.pre_train_path:
            Tools.print('==> Pre train from checkpoint {} ..'.format(self.pre_train_path))
            checkpoint = torch.load(self.pre_train_path)
            checkpoint_value = {key.replace("module.", "module.attention."): checkpoint["net"][key]
                                for key in checkpoint["net"].keys()}
            if 'acc' in checkpoint.keys() and 'epoch' in checkpoint.keys():
                best_acc = checkpoint['acc']
                best_epoch = checkpoint['epoch']
                Tools.print("{} {}".format(best_acc, best_epoch))
            net.load_state_dict(checkpoint_value, strict=False)
            pass

        # Load checkpoint.
        if self.resume:
            Tools.print('==> Resuming from checkpoint {} ..'.format(self.checkpoint_path))
            checkpoint = torch.load(self.checkpoint_path)
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            best_epoch = checkpoint['epoch']
            Tools.print("{} {}".format(best_acc, best_epoch))
            pass
        pass

    def test(self, epoch=0, is_test_test=True):
        self.net.eval()
        total = 0
        correct = 0
        loader = self.test_loader if is_test_test else self.train_loader
        for batch_idx, (inputs, targets, _) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = self.net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pass
        Tools.print("Epoch {} {} {}/({}-{})".format(epoch, " Test" if is_test_test else "Train",
                                                    correct/total, correct, total))
        return correct / total * 100

    def _train_one_epoch(self, epoch, test_per_epoch=3):
        # Train
        Tools.print()
        self.net.train()

        learning_rate = self._adjust_learning_rate(epoch)
        Tools.print("Epoch: [{}] lr={}".format(epoch, learning_rate))

        avg_loss = AverageMeter()
        for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer.zero_grad()

            out_logits = self.net(inputs)

            loss = self.criterion(out_logits, targets)
            avg_loss.update(loss.item(), inputs.size(0))
            loss.backward()
            self.optimizer.step()
            pass

        Tools.print('Epoch: [{}-{}] Loss: {avg_loss.val:.4f} ({avg_loss.avg:.4f})'.format(
            epoch, len(self.train_loader), avg_loss=avg_loss))

        # Test
        if epoch % test_per_epoch == 0:
            _ = self.test(epoch, is_test_test=False)
            _test_acc = self.test(epoch, is_test_test=True)

            if _test_acc > self.best_acc:
                Tools.print('Saving..')
                state = {'net': self.net.state_dict(), 'acc': _test_acc, 'epoch': epoch}
                torch.save(state, self.checkpoint_path)
                self.best_acc = _test_acc
                pass
            Tools.print('best accuracy: {:.2f}'.format(self.best_acc))
            pass
        pass

    def train(self, start_epoch, test_per_epoch=3):
        for epoch in range(start_epoch, self.max_epoch):
            self._train_one_epoch(epoch, test_per_epoch=test_per_epoch)
            pass
        pass

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    """
    0: 0.8269/0.8501  # 9_class_2048_norm_count_3level_512_128_lr_1000
    1: 0.8242/0.8467(0.8255/0.8445)
    2: 0.8644/0.8661(0.9090/0.8611)
    0: 0.8058/0.8255(0.8046/0.8221)
    1: 0.8215/0.8427(0.8223/0.8406)
    2: 0.8609/0.8665(0.9029/0.8620)
    
    0: 0.8203/0.8439(0.8170/0.8402)
    1: 0.8182/0.8427(0.8205/0.8376)
    2: 0.8909/0.8657(0.9012/0.8605)
    
    # fine tune(can draw a compare line)
    2: 0.9602/0.9263(0.9607/0.9255)  # classier_1024_2_0_0
    2: 0.9603/0.9282(0.9596/0.9265)  # classier_1024_2_1_1
    2: 0.9474/0.9294(0.9502/0.9275)  # classier_1024_0_1_1
    # 0.8536, 172, 11_class_1024_3level_256_64_500_no_memory_1_l1
    2: 0.9349  # classier_256_2_2_1
    2: 0.9358  # 0.8836 1361 + 1198, 11_class_1024_256_64_1600_no_32_1_l1_sum_0_321
    
    2: 0.8790  # 0.8358, 870, 9_class_1024_norm_count_3level_256_64_lr_1000, classier_1024_2_0_0
    2: 0.8787  # 0.8381, 898, 9_class_1024_norm_count_3level_256_64_lr_1000, classier_1024_2_0_0
    2: 0.8793  # 0.8406, 984, 9_class_1024_norm_count_3level_256_64_lr_1000, classier_1024_2_0_0
    2: 0.8770  # 0.8411, 977, 9_class_1024_norm_count_3level_256_64_lr_1000, classier_1024_2_1_0
    2: 0.8713  # 0.8386, 927, 9_class_1024_norm_count_3level_256_64_lr_1000, classier_256_2_2_0
    2: 0.8711  # 0.8386, 927, 9_class_1024_norm_count_3level_256_64_lr_1000, classier_64_2_4_0
    
    2: 0.8774  # 0.8411, 977, 9_class_2048_norm_count_3level_512_128_lr_1000, classier_2048_2_0_0
    2: 0.8782  # 0.8411, 977, 9_class_2048_norm_count_3level_512_128_lr_1000, classier_2048_2_1_0
    
    # 0.8469, 444, 9_class_1024_norm_count_3level_256_64_lr_500_no_memory
    2: 0.8771, classier_1024_2_0_0
    
    # 0.8505, 490, 11_class_1024_3level_256_64_500_no_memory_1_l1
    2: 0.8819 classier_1024_2_0_0
    2: 0.8830 classier_1024_2_1_0
    2: 0.8833 classier_256_2_2_0
    2: 0.8822 classier_256_2_3_0
    2: 0.8823 classier_64_2_4_0
    2: 0.8812 classier_64_2_5_0
    
    # 0.8536, 172, 11_class_1024_3level_256_64_500_no_memory_1_l1
    2: 0.8833 classier_1024_2_0_0
    2: 0.8844 classier_256_2_2_0
    
    # 0.8546, 317, 11_class_1024_3level_256_64_500_no_memory_1_l1
    2: 0.8858 classier_1024_2_0_0
    2: 0.8853 classier_1024_2_1_0
    2: 0.8867 classier_256_2_2_0
    2: 0.8878 classier_256_2_3_0
    2: 0.8843 classier_64_2_5_0
    2: 0.8844 classier_64_2_4_0
    
    # 0.8599, 958, 11_class_1024_3level_256_64_1000_no_memory_1_l1_sum
    2: 0.8862 classier_1024_2_0_0
    2: 0.8875 classier_1024_2_1_0
    2: 0.8861 classier_256_2_2_0
    2: 0.8847 classier_256_2_3_0
    2: 0.8870 classier_64_2_4_0
    2: 0.8848 classier_64_2_5_0
    
    # 0.862, 694, 11_class_1024_3level_256_64_1000_no_memory_1_l1_sum
    2: 0.8890 classier_1024_2_0_0
    2: 0.8899 classier_1024_2_1_0
    2: 0.8881 classier_256_2_2_0
    2: 0.8874 classier_256_2_3_0
    2: 0.8843 classier_64_2_4_0
    2: 0.8844 classier_64_2_5_0
    
    # 0.8724 1241, 11_class_1024_256_64_1600_no_32_1_l1_sum_1_321
    2: 0.9000 classier_1024_2_0_0
    2: 0.8993 classier_1024_2_1_0
    2: 0.9011 classier_256_2_2_0
    2: 0.8993 classier_256_2_3_0
    2: 0.8964 classier_64_2_4_0
    2: 0.8970 classier_64_2_5_0
    
    # 0.8759 1361, 11_class_1024_256_64_1600_no_32_1_l1_sum_0_321
    2: 0.9018 classier_1024_2_0_0
    2: 0.9012 classier_1024_2_1_0
    2: 0.8994 classier_256_2_2_0
    2: 0.8988 classier_256_2_3_0
    2: 0.8991 classier_64_2_4_0
    2: 0.8983 classier_64_2_5_0
    
    # 0.8836 1361 + 1198, 11_class_1024_256_64_1600_no_32_1_l1_sum_0_321
    2: 0.9049 classier_1024_2_0_0
    2: 0.9060 classier_1024_2_1_0
    2: 0.9048 classier_256_2_2_0
    2: 0.9047 classier_256_2_3_0
    2: 0.9041 classier_64_2_4_0
    2: 0.9028 classier_64_2_5_0
    
    # 0.8845 1361 + 1391, 11_class_1024_256_64_1600_no_32_1_l1_sum_0_321
    2: 0.9056 classier_1024_2_0_0
    2: 0.90 classier_1024_2_1_0
    2: 0.9043 classier_256_2_2_0
    2: 0.90 classier_256_2_3_0
    2: 0.90 classier_64_2_4_0
    2: 0.90 classier_64_2_5_0
    
    # 83.61, 11_class_1024_256_64_1600_no_32_1_l1_sum_0_555
    2: 0.9046 classier_1024_2_0_0
    2: 0.9066 classier_1024_2_1_0
    2: 0.9045 classier_256_2_2_0
    2: 0.9043 classier_256_2_3_0
    2: 0.8594 classier_64_2_4_0
    2: 0.8595 classier_64_2_5_0
    
    # 85.xx, 11_class_1024_256_64_1600_no_32_1_l1_sum_0_333
    2: 0.9059 classier_1024_2_0_0
    2: 0.9045 classier_1024_2_1_0
    2: 0.9041 classier_256_2_2_0
    2: 0.9048 classier_256_2_3_0
    2: 0.8858 classier_64_2_4_0
    2: 0.8859 classier_64_2_5_0
    """

    _which = 4
    _is_l2norm = True
    _is_fine_tune = False
    _classifier_type = 2  # 0, 1, 2

    # 1
    # _low_dim = [2048, 512, 128]
    # _name = "9_class_2048_norm_count_3level_512_128_lr_1000"
    # from cifar_9_class128_update_epoch_norm_count_3level import AttentionResNet

    # 2
    # _low_dim = [1024, 256, 64]
    # _name = "9_class_1024_norm_count_3level_256_64_lr_1000"
    # from cifar_9_class128_update_epoch_norm_count_3level import AttentionResNet

    # 3
    # _low_dim = [1024, 256, 64]
    # _name = "9_class_1024_norm_count_3level_256_64_lr_500_no_memory"
    # from cifar_9_class128_update_epoch_norm_count_3level import AttentionResNet

    # 4
    # _low_dim = [1024, 256, 64]
    # _name = "11_class_1024_3level_256_64_500_no_memory_1_l1"
    # from cifar_11_3level_no_memory_l2 import AttentionResNet

    # 5
    # _low_dim = [1024, 256, 64]
    # _name = "11_class_1024_3level_256_64_1000_no_memory_1_l1_sum"
    # from cifar_11_3level_no_memory_l2 import AttentionResNet

    # 6
    # _low_dim = [1024, 256, 64]
    # _name = "11_class_1024_256_64_1600_no_32_1_l1_sum_1_321"
    # from cifar_11_3level_no_memory_l2_sum import HCResNet as AttentionResNet

    # 7
    # _low_dim = [1024, 256, 64]
    # _name = "11_class_1024_256_64_1600_no_32_1_l1_sum_0_321"
    # from cifar_11_3level_no_memory_l2_sum import HCResNet as AttentionResNet

    # 8
    # _low_dim = [1024, 256, 64]
    # _name = "11_class_1024_256_64_1600_no_32_1_l1_sum_0_333"
    # from cifar_11_3level_no_memory_l2_sum import HCResNet as AttentionResNet

    # 9
    _low_dim = [1024, 512, 256, 128, 64]
    _name = "11_class_1024_5level_512_256_128_64_no_1600_32_1_l1_sum_0_54321"
    from cifar_10_5level_z import HCResNet as AttentionResNet

    _which_out = _which * 2 + (1 if _is_l2norm else 0)
    _input_size = _low_dim[_which]  # first input size

    _start_epoch = 0  # train epoch
    _max_epoch = 200
    _linear_bias = False

    # _pre_train_path = None
    _pre_train_path = "./checkpoint/{}/ckpt.t7".format(_name)
    _checkpoint_path_classier = "./checkpoint/{}/classier_{}_{}_{}_{}.t7".format(
        _name, _input_size, _classifier_type, _which_out, 1 if _is_fine_tune else 0)

    Tools.print()
    Tools.print("input_size={} name={}".format(_input_size, _name))
    Tools.print("classier={}".format(_checkpoint_path_classier))

    _net = Classifier(input_size_or_list=_input_size if _classifier_type == 0 else [_input_size, 512, 256],
                      low_dim=_low_dim, classifier_type=_classifier_type,
                      which_out=_which_out, is_fine_tune=_is_fine_tune, linear_bias=_linear_bias)
    runner = ClassierRunner(net=_net, max_epoch=_max_epoch, resume=False, is_fine_tune=_is_fine_tune,
                            pre_train_path=_pre_train_path, checkpoint_path=_checkpoint_path_classier)
    Tools.print()
    train_acc = runner.test(0, is_test_test=False)
    test_acc = runner.test(0, is_test_test=True)
    Tools.print('Random accuracy: {:.2f}/{:.2f}'.format(train_acc, test_acc))
    runner.train(start_epoch=_start_epoch, test_per_epoch=1)
    Tools.print()
    train_acc = runner.test(0, is_test_test=False)
    test_acc = runner.test(0, is_test_test=True)
    Tools.print('final accuracy: {:.2f}/{:.2f}'.format(train_acc, test_acc))
    pass
