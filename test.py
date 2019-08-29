import os
import re
import math
import torch
import resnet
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils import data
import torch.nn.functional as F
from PIL import Image, ImageOps
from collections import namedtuple
from alisuretool.Tools import Tools
from torch.nn.functional import upsample
import torchvision.transforms as transform
from torch.nn.parallel.data_parallel import DataParallel


Split = namedtuple("split", ["train", "val", "test_val"])("train", "val", "test_val")


class CityscapesDataset(data.Dataset):

    num_class = 19

    def __init__(self, root_dir='../datasets/cityscapes', split=Split.train, is_train=True,
                 transform=None, base_size=520, crop_size=480, scale=True):
        self.split = split
        self.is_train = is_train
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.scale = scale

        self.images, self.masks = self._get_pairs(root_dir, split)
        pass

    @staticmethod
    def _get_pairs(folder, split):
        img_paths, mask_paths = [], []
        split_f = os.path.join(folder, 'train_fine.txt' if split == Split.train else 'val_fine.txt')
        with open(split_f, 'r') as lines:
            for line in lines:
                ll_str = re.split('\t', line)
                imgpath = os.path.join(folder, ll_str[0].rstrip())
                maskpath = os.path.join(folder, ll_str[1].rstrip())
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    Tools.print('cannot find the mask: {}'.format(maskpath))
                pass
            pass
        # return img_paths[0: 10], mask_paths[0: 10]
        return img_paths, mask_paths

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])

        if self.is_train:
            if self.split == Split.train:
                img, mask = self._transform_mirror_resize_pad_and_crop(img, mask)
            else:
                img, mask = self._transform_resize_and_crop(img, mask)
            pass

        if self.transform is not None:
            img = self.transform(img)
            pass

        mask = self._mask_transform(mask)
        return img, mask

    @classmethod
    def read_image(cls, image_path, transform):
        img = Image.open(image_path).convert('RGB')
        img = cls._transform_resize_and_crop(img)
        img = transform(img)
        return img

    @staticmethod
    def _transform_resize_and_crop(img, mask=None, crop_size=768):
        w, h = img.size
        if w > h:
            oh = crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = crop_size
            oh = int(1.0 * h * ow / w)

        img = img.resize((ow, oh), Image.BILINEAR)

        # center crop
        w, h = img.size
        x1 = int(round((w - crop_size) / 2.))
        y1 = int(round((h - crop_size) / 2.))
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        if mask:
            mask = mask.resize((ow, oh), Image.NEAREST)
            mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
            return img, mask
        return img

    def _transform_mirror_resize_pad_and_crop(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            pass

        short_size = random.randint(int(self.base_size * 0.75),
                                    int(self.base_size * 2.0)) if self.scale else self.base_size
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # random rotate -10~10, mask using NN rotate
        # deg = random.uniform(-10, 10)
        # img = img.rotate(deg, resample=Image.BILINEAR)
        # mask = mask.rotate(deg, resample=Image.NEAREST)

        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)  # pad 255 for cityscapes
            pass

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        # gaussian blur as in PSP
        # if random.random() < 0.5:
        #     img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        # final transform
        return img, mask

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    pass


class Metrics(object):

    @staticmethod
    def batch_pix_accuracy(predict, target):
        _, predict = torch.max(predict, 1)
        predict = predict.cpu().numpy() + 1
        target = target.cpu().numpy() + 1
        pixel_labeled = np.sum(target > 0)
        pixel_correct = np.sum((predict == target) * (target > 0))
        assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
        return pixel_correct, pixel_labeled

    @staticmethod
    def batch_intersection_union(predict, target, nclass):
        _, predict = torch.max(predict, 1)
        mini = 1
        maxi = nclass
        nbins = nclass
        predict = predict.cpu().numpy() + 1
        target = target.cpu().numpy() + 1

        predict = predict * (target > 0).astype(predict.dtype)
        intersection = predict * (predict == target)
        # areas of intersection and union
        area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
        area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
        area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
        area_union = area_pred + area_lab - area_inter
        assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
        return area_inter, area_union

    @staticmethod
    def pixel_accuracy(im_pred, im_lab):
        im_pred = np.asarray(im_pred)
        im_lab = np.asarray(im_lab)

        pixel_labeled = np.sum(im_lab > 0)
        pixel_correct = np.sum((im_pred == im_lab) * (im_lab > 0))
        # pixel_accuracy = pixel_correct / pixel_labeled
        return pixel_correct, pixel_labeled

    @staticmethod
    def intersection_and_union(im_pred, im_lab, num_class):
        im_pred = np.asarray(im_pred)
        im_lab = np.asarray(im_lab)
        # Remove classes from unlabeled pixels in gt image.
        im_pred = im_pred * (im_lab > 0)
        # Compute area intersection:
        intersection = im_pred * (im_pred == im_lab)
        area_inter, _ = np.histogram(intersection, bins=num_class - 1, range=(1, num_class - 1))
        # Compute area union:
        area_pred, _ = np.histogram(im_pred, bins=num_class - 1, range=(1, num_class - 1))
        area_lab, _ = np.histogram(im_lab, bins=num_class - 1, range=(1, num_class - 1))
        area_union = area_pred + area_lab - area_inter
        return area_inter, area_union

    @staticmethod
    def _get_voc_pallete(num_cls):
        n = num_cls
        pallete = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while (lab > 0):
                pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i = i + 1
                lab >>= 3
        return pallete

    adepallete = [0, 0, 0, 120, 120, 120, 180, 120, 120, 6, 230, 230, 80, 50, 50, 4, 200, 3, 120, 120, 80, 140, 140,
                  140, 204, 5, 255, 230, 230, 230, 4, 250, 7, 224, 5, 255, 235, 255, 7, 150, 5, 61, 120, 120, 70, 8,
                  255, 51, 255, 6, 82, 143, 255, 140, 204, 255, 4, 255, 51, 7, 204, 70, 3, 0, 102, 200, 61, 230, 250,
                  255, 6, 51, 11, 102, 255, 255, 7, 71, 255, 9, 224, 9, 7, 230, 220, 220, 220, 255, 9, 92, 112, 9, 255,
                  8, 255, 214, 7, 255, 224, 255, 184, 6, 10, 255, 71, 255, 41, 10, 7, 255, 255, 224, 255, 8, 102, 8,
                  255, 255, 61, 6, 255, 194, 7, 255, 122, 8, 0, 255, 20, 255, 8, 41, 255, 5, 153, 6, 51, 255, 235, 12,
                  255, 160, 150, 20, 0, 163, 255, 140, 140, 140, 250, 10, 15, 20, 255, 0, 31, 255, 0, 255, 31, 0, 255,
                  224, 0, 153, 255, 0, 0, 0, 255, 255, 71, 0, 0, 235, 255, 0, 173, 255, 31, 0, 255, 11, 200, 200, 255,
                  82, 0, 0, 255, 245, 0, 61, 255, 0, 255, 112, 0, 255, 133, 255, 0, 0, 255, 163, 0, 255, 102, 0, 194,
                  255, 0, 0, 143, 255, 51, 255, 0, 0, 82, 255, 0, 255, 41, 0, 255, 173, 10, 0, 255, 173, 255, 0, 0, 255,
                  153, 255, 92, 0, 255, 0, 255, 255, 0, 245, 255, 0, 102, 255, 173, 0, 255, 0, 20, 255, 184, 184, 0, 31,
                  255, 0, 255, 61, 0, 71, 255, 255, 0, 204, 0, 255, 194, 0, 255, 82, 0, 10, 255, 0, 112, 255, 51, 0,
                  255, 0, 194, 255, 0, 122, 255, 0, 255, 163, 255, 153, 0, 0, 255, 10, 255, 112, 0, 143, 255, 0, 82, 0,
                  255, 163, 255, 0, 255, 235, 0, 8, 184, 170, 133, 0, 255, 0, 255, 92, 184, 0, 255, 255, 0, 31, 0, 184,
                  255, 0, 214, 255, 255, 0, 112, 92, 255, 0, 0, 224, 255, 112, 224, 255, 70, 184, 160, 163, 0, 255, 153,
                  0, 255, 71, 255, 0, 255, 0, 163, 255, 204, 0, 255, 0, 143, 0, 255, 235, 133, 255, 0, 255, 0, 235, 245,
                  0, 255, 255, 0, 122, 255, 245, 0, 10, 190, 212, 214, 255, 0, 0, 204, 255, 20, 0, 255, 255, 255, 0, 0,
                  153, 255, 0, 41, 255, 0, 255, 204, 41, 0, 255, 41, 255, 0, 173, 0, 255, 0, 245, 255, 71, 0, 255, 122,
                  0, 255, 0, 255, 184, 0, 92, 255, 184, 255, 0, 0, 133, 255, 255, 214, 0, 25, 194, 194, 102, 255, 0, 92,
                  0, 255]

    citypallete = [
        128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30, 220, 220, 0,
        107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100, 0, 80, 100,
        0, 0, 230, 119, 11, 32, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0, 192, 64,
        0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0,
        128, 64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0, 64, 192, 0, 64, 64, 128, 64,
        192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128,
        192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64, 64, 64, 192, 64, 192, 192,
        64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0,
        128, 160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0,
        128, 96, 128, 128, 224, 128, 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32,
        192, 128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64, 128, 96, 192,
        128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192,
        160, 128, 192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0, 192, 96, 128, 192, 224,
        128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192, 192, 160, 192,
        192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192, 96, 192, 192, 224, 192, 192,
        0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0,
        192, 32, 0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192, 160, 128, 0, 96, 0, 128, 96,
        0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224,
        0, 192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128, 32, 64, 0, 160, 64, 128,
        160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160,
        64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64, 128, 224, 64, 0,
        96, 192, 128, 96, 192, 0, 224, 192, 128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96,
        192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160, 0, 32, 32, 128,
        160, 32, 128, 32, 160, 128, 160, 160, 128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32,
        128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128, 32,
        224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224,
        128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192, 32, 160, 192,
        160, 160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224,
        160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224, 192, 160, 224,
        192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 0, 0, 0]

    @classmethod
    def get_mask_pallete(cls, npimg, dataset='detail'):
        if dataset == 'pascal_voc':
            npimg[npimg == 21] = 255
        # put colormap
        out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
        if dataset == 'ade20k':
            out_img.putpalette(cls.adepallete)
        elif dataset == 'cityscapes':
            out_img.putpalette(cls.citypallete)
        else:
            out_img.putpalette(cls._get_voc_pallete(256))
        return out_img

    pass


class PAMModule(nn.Module):

    def __init__(self, in_dim):
        super(PAMModule, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        pass

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

    pass


class CAMModule(nn.Module):

    def __init__(self, in_dim):
        super(CAMModule, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        pass

    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

    pass


class DANetHead(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels), nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels), nn.ReLU())

        self.sa = PAMModule(inter_channels)
        self.sc = CAMModule(inter_channels)

        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels), nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels), nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
        pass

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        output = [sasc_output, sa_output, sc_output]
        return tuple(output)

    def __call__(self, *args, **kwargs):
        return super(DANetHead, self).__call__(*args, **kwargs)

    pass


class DANet(nn.Module):

    def __init__(self, n_class, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d,
                 base_size=576, crop_size=608, mean=list([.485, .456, .406]), std=list([.229, .224, .225]),
                 multi_grid=False, multi_dilation=None, dilated=True):
        super(DANet, self).__init__()

        self.n_class = n_class
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size

        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(dilated=dilated, norm_layer=norm_layer,
                                              multi_grid=multi_grid, multi_dilation=multi_dilation)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(dilated=dilated, norm_layer=norm_layer,
                                               multi_grid=multi_grid, multi_dilation=multi_dilation)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(dilated=dilated, norm_layer=norm_layer,
                                               multi_grid=multi_grid, multi_dilation=multi_dilation)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}

        self.head = DANetHead(2048, n_class, norm_layer)
        pass

    def base_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c4

    def forward(self, x):
        im_size = x.size()[2:]

        c4 = self.base_forward(x)
        print(c4.shape)
        x = self.head(c4)

        x = list(x)
        x[0] = upsample(x[0], im_size, **self._up_kwargs)
        x[1] = upsample(x[1], im_size, **self._up_kwargs)
        x[2] = upsample(x[2], im_size, **self._up_kwargs)

        outputs = [x[0], x[1], x[2]]
        return tuple(outputs)

    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = Metrics.batch_pix_accuracy(pred.data, target.data)
        inter, union = Metrics.batch_intersection_union(pred.data, target.data, self.n_class)
        return correct, labeled, inter, union

    pass


class MultiEvalModule(DataParallel):

    def __init__(self, module, n_class, device_ids=None, flip=True, multi_scales=False):
        super(MultiEvalModule, self).__init__(module, device_ids)
        self.n_class = n_class
        self.flip = flip
        self.base_size = module.base_size
        self.crop_size = module.crop_size
        self.scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2] if multi_scales else [1.0]
        pass

    def parallel_forward(self, inputs):
        inputs = [(input.unsqueeze(0).cuda(device),) for input, device in zip(inputs, self.device_ids)]
        replicas = self.replicate(self, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, [{} for _ in range(len(inputs))])
        return outputs

    def forward(self, image):
        batch, _, h, w = image.size()
        assert(batch == 1)
        stride_rate = 2.0 / 3.0 if len(self.scales) == 1 else 1.0/2.0

        crop_size = self.crop_size
        stride = int(crop_size * stride_rate)
        with torch.cuda.device_of(image):
            scores = image.new().resize_(batch, self.n_class, h, w).zero_().cuda()

        for scale in self.scales:
            long_size = int(math.ceil(self.base_size * scale))

            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)
                short_size = height
                pass

            # resize image to current size
            cur_img = self.resize_image(image, height, width, **self.module._up_kwargs)

            if long_size <= crop_size:
                pad_img = self.pad_image(cur_img, self.module.mean, self.module.std, crop_size)
                outputs = self.module_inference(self.module, pad_img, self.flip)
                outputs = self.crop_image(outputs, 0, height, 0, width)
            else:
                pad_img = self.pad_image(cur_img, self.module.mean,
                                         self.module.std, crop_size) if short_size < crop_size else cur_img
                _, _, ph, pw = pad_img.size()
                assert(ph >= height and pw >= width)

                # grid forward and normalize
                h_grids = int(math.ceil(1.0 * (ph-crop_size)/stride)) + 1
                w_grids = int(math.ceil(1.0 * (pw-crop_size)/stride)) + 1
                with torch.cuda.device_of(image):
                    outputs = image.new().resize_(batch,self.n_class,ph,pw).zero_().cuda()
                    count_norm = image.new().resize_(batch,1,ph,pw).zero_().cuda()
                    pass

                # grid evaluation
                for idh in range(h_grids):
                    for idw in range(w_grids):
                        h0, w0 = idh * stride, idw * stride
                        h1, w1 = min(h0 + crop_size, ph), min(w0 + crop_size, pw)
                        crop_img = self.crop_image(pad_img, h0, h1, w0, w1)
                        pad_crop_img = self.pad_image(crop_img, self.module.mean, self.module.std, crop_size)
                        output = self.module_inference(self.module, pad_crop_img, self.flip)
                        outputs[:,:,h0:h1,w0:w1] += self.crop_image(output, 0, h1-h0, 0, w1-w0)
                        count_norm[:,:,h0:h1,w0:w1] += 1
                        pass
                    pass

                assert((count_norm==0).sum() == 0)
                outputs = outputs / count_norm
                outputs = outputs[:,:,:height,:width]
                pass

            score = self.resize_image(outputs, h, w, **self.module._up_kwargs)
            scores += score

        return scores

    @classmethod
    def module_inference(cls, module, image, flip=True):
        output = module.evaluate(image)
        if flip:
            fimg = cls.flip_image(image)
            foutput = module.evaluate(fimg)
            output += cls.flip_image(foutput)
        return output.exp()

    @staticmethod
    def resize_image(img, h, w, **up_kwargs):
        return F.upsample(img, (h, w), **up_kwargs)

    @staticmethod
    def pad_image(img, mean, std, crop_size):
        b, c, h, w = img.size()
        assert (c == 3)
        padh = crop_size - h if h < crop_size else 0
        padw = crop_size - w if w < crop_size else 0
        pad_values = -np.array(mean) / np.array(std)
        img_pad = img.new().resize_(b, c, h + padh, w + padw)
        for i in range(c):
            # note that pytorch pad params is in reversed orders
            img_pad[:, i, :, :] = F.pad(img[:, i, :, :], (0, padw, 0, padh), value=pad_values[i])
        assert (img_pad.size(2) >= crop_size and img_pad.size(3) >= crop_size)
        return img_pad

    @staticmethod
    def crop_image(img, h0, h1, w0, w1):
        return img[:, :, h0:h1, w0:w1]

    @staticmethod
    def flip_image(img):
        assert (img.dim() == 4)
        with torch.cuda.device_of(img):
            idx = torch.arange(img.size(3) - 1, -1, -1).type_as(img).long()
        return img.index_select(3, idx)

    pass


class Tester(object):

    def __init__(self, checkpoint_file, batch_size, base_size=2048, crop_size=768,
                 multi_scales=False, is_train=False, is_simple=True):
        self.checkpoint_file = checkpoint_file
        self.batch_size = batch_size
        self.base_size = base_size
        self.crop_size = crop_size
        self.multi_scales = multi_scales
        self.is_train = is_train
        self.is_simple = is_simple

        self.input_transform = transform.Compose([transform.ToTensor(),
                                                  transform.Normalize([.485, .456, .406], [.229, .224, .225])])

        self.train_set, self.train_loader = self._get_data(split=Split.train, is_train=self.is_train)
        self.test_val_set, self.test_val_loader = self._get_data(split=Split.test_val, is_train=self.is_train)
        self.val_set, self.val_loader = self._get_data(split=Split.val, is_train=self.is_train)

        self.num_class = self.train_set.num_class

        self.model = DANet(self.num_class, backbone="resnet101", aux=False, se_loss=False, norm_layer=nn.BatchNorm2d,
                           base_size=base_size, crop_size=crop_size, multi_grid=True, multi_dilation=[4, 8, 16])

        self._load_model()

        if self.is_simple:
            self.model_eval = nn.DataParallel(self.model).cuda()
        else:
            self.model_eval = MultiEvalModule(self.model, self.num_class, multi_scales=self.multi_scales).cuda()

        pass

    def _get_data(self, split, is_train):
        data_set = CityscapesDataset(split=split, is_train=is_train, transform=self.input_transform,
                                     base_size=self.base_size, crop_size=self.crop_size, scale=True)
        data_loader = data.DataLoader(data_set, self.batch_size, drop_last=False, shuffle=False, num_workers=2)
        return data_set, data_loader

    def _load_model(self):
        if self.checkpoint_file:
            Tools.print("resume file is {}, ".format(self.checkpoint_file))
            checkpoint = torch.load(self.checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            pass
        pass

    def _eval_batch(self, image, dst):
        outputs = self.model_eval(image) if self.is_simple else self.model_eval.parallel_forward(image)
        batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
        for output, target in zip(outputs, dst):
            correct, labeled = Metrics.batch_pix_accuracy(output.data.cpu(), target)
            inter, union = Metrics.batch_intersection_union(output.data.cpu(), target, self.num_class)
            batch_correct += correct
            batch_label += labeled
            batch_inter += inter
            batch_union += union
        return batch_correct, batch_label, batch_inter, batch_union

    def eval(self, test_loader, split=Split.val):
        self.model_eval.eval()

        pixAcc, mIoU, IoU = 0, 0, 0
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        for i, (image, target) in enumerate(test_loader):
            with torch.no_grad():
                correct, labeled, inter, union = self._eval_batch(image, target)
                total_correct += correct.astype('int64')
                total_label += labeled.astype('int64')
                total_inter += inter.astype('int64')
                total_union += union.astype('int64')
                pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
                IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
                mIoU = IoU.mean()
                Tools.print('pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
            pass

        Tools.print("================ {} IOU ================".format(split))
        Tools.print("Mean IoU over %d classes: %.4f" % (self.num_class, mIoU))
        Tools.print("Pixel-wise Accuracy: %2.2f%%" % (pixAcc * 100))
        return pixAcc, mIoU, IoU

    def vis_one(self, im_path, out_dir, dataset="cityscapes"):
        self.model_eval.eval()

        image = CityscapesDataset.read_image(im_path, transform=self.input_transform)
        image = torch.from_numpy(np.expand_dims(image, axis=0))
        outputs = self.model_eval(image) if self.is_simple else self.model_eval.parallel_forward(image)
        predicts = [torch.max(output, 1)[1].cpu().numpy() for output in outputs]
        mask = Metrics.get_mask_pallete(predicts[0], dataset)
        outname = os.path.splitext(os.path.basename(im_path))[0] + '.png'
        mask.save(os.path.join(out_dir, outname))
        pass

    pass


if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    _batch_size = 2
    _checkpoint_file = "../datasets/DANet101.pth.tar"

    tester = Tester(_checkpoint_file, _batch_size, is_simple=False)

    tester.vis_one(im_path=tester.test_val_set.images[0], out_dir=Tools.new_dir("./cityscapes/vis"))
    # tester.eval(tester.val_loader, split=Split.val)
    tester.eval(tester.test_val_loader, split=Split.test_val)
    # tester.eval(tester.train_loader, split=Split.train)

    print('Evaluation is finished!!!')
    pass

