"""
 * coco
 * Created on 03.04.19
 * Author: doering
"""
import torch
import numpy as np
import json
import cv2 as cv
from torch.utils.data import Dataset
from data_utils import apply_augmentation_test
import matplotlib.pyplot as plt
from pycocotools import mask


class GenerateHeatmap:
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        hms = np.zeros(shape = (self.num_parts, self.output_res, self.output_res), dtype = np.float32)
        sigma = self.sigma
        for p in keypoints:
                    x, y = int(p[0]/4), int(p[1]/4)
                    if x<0 or y<0 or x>=self.output_res or y>=self.output_res:
                        continue
                    ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                    br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                    c,d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a,b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc,dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa,bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[0, aa:bb,cc:dd] = np.maximum(hms[0, aa:bb,cc:dd], self.g[a:b,c:d])
        return hms


class Coco(Dataset):

    def __init__(self, file_path, dtype, output_size, transform=None):

        self.flip_ref = [i - 1 for i in [1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16]]
        self.dtype = dtype
        self.is_train = (dtype == 'train')
        self.transform = transform
        self.output_size = output_size
        self.heatmapres = output_size // 4
        self.generate_heatmaps = GenerateHeatmap(self.heatmapres, 1)

        with open(file_path) as anno_file:
            self.anno = json.load(anno_file)

    def __len__(self):

        length = len(self.anno)

        return length

    def apply_augmentation(self, example, is_train, output_size):

        im = cv.imread(example[0]['image'], 1)
        height, width = im.shape[:2]

        crop_pos = [width // 2, height // 2]
        max_d = max(height, width)

        scales = [output_size / float(max_d), output_size / float(max_d)]
        centers = []
        corners = []

        m = np.zeros((height, width))
        for j in example:
            if j['iscrowd']:
                rle = mask.frPyObjects(j['mask'], height, width)
                m += mask.decode(rle)
        m = m < 0.5
        m = m.astype(np.float32)

        for ann in example:
            if ann['iscrowd']:
                continue
            poly = ann['mask']
            if len(poly[0]) < 25:
                continue
            N = len(poly[0])
            X = poly[0][0:N:2]
            Y = poly[0][1:N:2]
            c = np.ones((3, 1))
            x1, y1 = ann['bbox'][0], ann['bbox'][1]
            w, h = ann['bbox'][2], ann['bbox'][3]
            c[0], c[1] = x1 + w / 2, y1 + h / 2
            centers.append(c)
            corner = np.ones((3, len(X)))
            corner[0, :] = X
            corner[1, :] = Y
            corners.append(corner)

        param = {'rot': 0,
                 'scale': scales[0],  # scale,
                 'flip': 0,
                 'tx': 0,
                 'ty': 0}

        if is_train:
            np.random.seed()
            param['scale'] = scales[0] * (np.random.random() + .5)  # scale * (np.random.random() + .5)
            param['flip'] = np.random.binomial(1, 0.5)
            param['rot'] = (np.random.random() * (40 * 0.0174532)) - 20 * 0.0174532
            param['tx'] = np.int8((np.random.random() * 10) - 5)
            param['ty'] = np.int8((np.random.random() * 10) - 5)

        a = param['scale'] * np.cos(param['rot'])
        b = param['scale'] * np.sin(param['rot'])

        shift_to_upper_left = np.identity(3)
        shift_to_center = np.identity(3)

        t = np.identity(3)
        t[0][0] = a
        if param['flip']:
            t[0][0] = -a

        t[0][1] = -b
        t[1][0] = b
        t[1][1] = a

        shift_to_upper_left[0][2] = -crop_pos[0] + param['tx']
        shift_to_upper_left[1][2] = -crop_pos[1] + param['ty']
        shift_to_center[0][2] = output_size / 2
        shift_to_center[1][2] = output_size / 2
        t_form = np.matmul(t, shift_to_upper_left)
        t_form = np.matmul(shift_to_center, t_form)

        centers_warped = []
        corners_warped = []

        im_cv = cv.warpAffine(im, t_form[0:2, :], (output_size, output_size))

        for i, c in enumerate(centers):
            ct = np.matmul(t_form, c)
            cr = np.matmul(t_form, corners[i])
            cr[0, :] *= cr[0, :] > -1
            cr[0, :] *= cr[0, :] < output_size
            cr[1, :] *= cr[1, :] > 0
            cr[1, :] *= cr[1, :] < output_size

            x1, x2 = min(cr[0, :]), max(cr[0, :])
            y1, y2 = min(cr[1, :]), max(cr[1, :])
            if x1 == -1 or y1 == -1 or x2 >= output_size or y2 >= output_size:
                print("prob")
            w = np.abs(x1 - x2)
            h = np.abs(y1 - y2)
            m_dim = np.maximum(w, h)
            if m_dim == 0:
                continue
            corners_warped.append(np.log(m_dim))
            centers_warped.append(ct)

        mw = cv.warpAffine(m*255, t_form[0:2, :], (output_size, output_size))/255
        mw = cv.resize(mw, (self.heatmapres, self.heatmapres))
        mw = (mw > 0.5).astype(np.float32)
        hms = self.generate_heatmaps(centers_warped)
        votes, masks = self.generate_offsets(centers_warped, corners_warped)

        img = cv.cvtColor(im_cv, cv.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float()
        img = torch.transpose(img, 1, 2)
        img = torch.transpose(img, 0, 1)
        img /= 255
        hms = torch.from_numpy(hms).float()
        mw = torch.from_numpy(mw)
        mw = mw.view(1, self.heatmapres, self.heatmapres)

        return img, hms, mw, votes, masks

    def generate_offsets(self, keypoints, offsets):

        output_res = self.heatmapres
        srmasks = torch.zeros(1, output_res, output_res)
        sroff = torch.zeros(1, output_res, output_res)

        for i, p in enumerate(keypoints):
            x, y = int(p[0] / 4), int(p[1] / 4)
            if x < 0 or y < 0 or x >= output_res or y >= output_res:
                continue
            yrs = range(max(0, int(y) - 2), min(63, int(y) + 3))
            xrs = range(max(0, int(x) - 2), min(63, int(x) + 3))
            for xr in xrs:
                for yr in yrs:
                    #yrim = yr * 4
                    #xrim = xr * 4
                    #rx = xim - xrim
                    #ry = yim - yrim
                    sroff[0][yr][xr] = offsets[i]
                    #sroff[1][yr][xr] = offsets[i][0][1]  - #offsets[i][1]/256
                    srmasks[0][yr][xr] = 1  # rx
                    #srmasks[1][yr][xr] = 1  # ry#

        return sroff, srmasks

    def __getitem__(self, idx):

        example = self.anno[idx]

        if self.dtype == 'train':

            return self.apply_augmentation(example, self.dtype, self.output_size)
        else:
            images, imagesf, warps = apply_augmentation_test(example, output_size=self.output_size)
            meta = {'img': example[0]['image'], 'imgID': example[0]['im_id'],
                    'warps': warps, 'L': len(example)}

            return images, imagesf, meta#{'images': images, 'imagesf': imagesf, 'meta': meta}