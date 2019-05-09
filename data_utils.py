import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch import nn


def compute_OKS(gt, dt, bb, area):

    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    vars = (sigmas * 2)**2
    k = len(sigmas)
    g = np.array(gt)
    xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
    k1 = np.count_nonzero(vg > 0)
    x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
    y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
    d = np.array(dt)
    xd = d[0::3]; yd = d[1::3]
    if k1>0:
        # measure the per-keypoint distance if keypoints visible
        dx = xd - xg
        dy = yd - yg
    else:
        # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
        z = np.zeros((k))
        dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
        dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
    e = (dx**2 + dy**2) / vars / (area.item()+np.spacing(1)) / 2
    if k1 > 0:
        e=e[vg > 0]
    OKS = np.sum(np.exp(-e)) / e.shape[0]

    return OKS


def plot_masks_on_image(im, masks):

      n_masks = masks.shape[0]
      #im = torch.transpose(im, 1, 2)
      #im = torch.transpose(im, 0, 1)
      for j in range(n_masks):

          r = cv2.resize(masks[j].numpy(), (256, 256))
          im[0, :, :] = im[0, :, :] + torch.from_numpy(r)

      return im

def get_transform(param, crop_pos, output_size, scales):
    shift_to_upper_left = np.identity(3)
    shift_to_center = np.identity(3)

    a = scales[0] * param['scale'] * np.cos(param['rot'])
    b = scales[1] * param['scale'] * np.sin(param['rot'])

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

    return t_form


def apply_augmentation_test(example, output_size=256):
    im = cv2.imread(example, 1)
    height, width = im.shape[:2]

    crop_pos = [width / 2, height / 2]
    max_d = max(height, width)
    scales = [output_size / float(max_d), output_size / float(max_d)]

    param = {'rot': 0,
             'scale': 1,
             'flip': 0,
             'tx': 0,
             'ty': 0}

    t_form = get_transform(param, crop_pos, output_size, scales)
    im_cv = cv2.warpAffine(im, t_form[0:2, :], (output_size, output_size))
    img = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
    imf = cv2.flip(img, 1)

    img = torch.from_numpy(img).float()
    img = torch.transpose(img, 1, 2)
    img = torch.transpose(img, 0, 1)
    img /= 255

    imf = torch.from_numpy(imf).float()
    imf = torch.transpose(imf, 1, 2)
    imf = torch.transpose(imf, 0, 1)
    imf /= 255

    warp = torch.from_numpy(np.linalg.inv(t_form))

    return img, imf, warp


def get_preds(prs, scales, warp):

    pool = nn.MaxPool2d(3, 1, 1).cuda()
    o = pool(prs.cuda()).data.cpu()
    maxm = torch.eq(o, prs).float()
    prs = prs * maxm
    res = 64
    prso = prs.view(res * res)
    val_k, ind = prso.topk(30, dim=0)
    xs = ind % res
    ys = (ind / res).long()
    xst, yst, sc = [], [], []
    N = len(val_k)
    for i in range(N):
        if val_k[i] >= 0.15:
            xst.append(xs[i].item() * 4)
            yst.append(ys[i].item() * 4)
            sc.append(np.exp(scales[0][ys[i]][xs[i]]) * warp[0][0])

    points = np.ones((3, len(sc)))
    points[0, :], points[1, :] = xst, yst
    dets = np.matmul(warp, points)

    return dets, sc


def apply_augmentation_test_td(path, example, output_size=256):

    im = cv2.imread(path, 1)
    crop_pos = example['center']
    max_d = example['scale']
    scales = [output_size / float(max_d), output_size / float(max_d)]

    param = {'rot': 0,
             'scale': 1,
             'flip': 0,
             'tx': 0,
             'ty': 0}

    t_form = get_transform(param, crop_pos, output_size, scales)
    im_cv = cv2.warpAffine(im, t_form[0:2, :], (output_size, output_size))
    img = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
    imf = cv2.flip(img, 1)

    img = torch.from_numpy(img).float()
    img = torch.transpose(img, 1, 2)
    img = torch.transpose(img, 0, 1)
    img /= 255

    imf = torch.from_numpy(imf).float()
    imf = torch.transpose(imf, 1, 2)
    imf = torch.transpose(imf, 0, 1)
    imf /= 255

    warp = torch.from_numpy(np.linalg.inv(t_form))

    return img, imf, warp


def get_preds_td(prs, mat, sr):

    pool = nn.MaxPool2d(3, 1, 1).cuda()

    xoff = sr[0:17]
    yoff = sr[17:34]

    prs2 = prs

    o = pool(prs.cuda()).data.cpu()
    maxm = torch.eq(o, prs).float()
    prs = prs * maxm
    res = 64
    prso = prs.view(17, res * res)
    val_k, ind = prso.topk(1, dim=1)
    xs = ind % res
    ys = (ind / res).long()


    keypoints = []
    score = 0
    points = torch.zeros(17, 2)
    c = 0

    for j in range(17):

        x, y = xs[j][0], ys[j][0]
        dx = xoff[j][int(y)][int(x)]
        dy = yoff[j][int(y)][int(x)]
        points[j][0] = (x * 4) + dx.item()
        points[j][1] = (y * 4) + dy.item()

        score += val_k[j][0]
        c += 1

    score /= c

    for j in range(17):

        point = torch.ones(3, 1)
        #if points[j][0] > 0 and points[j][1] > 0:
        point[0][0] = points[j][0]
        point[1][0] = points[j][1]
        #else:
        #    point[0][0] = xm
        #    point[1][0] = ym

        keypoint = np.matmul(mat, point)
        keypoints.append(float(keypoint[0].item()))
        keypoints.append(float(keypoint[1].item()))
        #keypoints.append(int(point[0][0]))
        #keypoints.append(int(point[1][0]))
        keypoints.append(1)

    return keypoints, score.item()
