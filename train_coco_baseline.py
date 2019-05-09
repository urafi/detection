"""
 * train_coco_baseline
 * Created on 03.04.19
 * Author: doering
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import random
import time
import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from bn_inception import bninception
from coco import Coco

from tqdm import tqdm
from argparser import parse_args
import matplotlib.pyplot as plt


def train(data_loader, model, optimizer):

    model.train()
    #total_loss = 0
    start_time = time.time()
    det_loss = 0
    scale_loss = 0

    for i, (img, det_maps, wts, votes, wvotes) in enumerate(tqdm(data_loader)):

        dt_target = det_maps.cuda(non_blocking=True)
        inputs = img.cuda(non_blocking=True)
        wts = wts.cuda(non_blocking=True)
        votes = votes.cuda(non_blocking=True)
        wvotes = wvotes.cuda(non_blocking=True)

        output = model(inputs)

        N = len(output)
        l1 = []
        l2 = []
        for n in range(N):
            output_det = output[n][:, 0:1]
            output_votes = output[n][:, 1:2] * wvotes
            l1.append(F.binary_cross_entropy_with_logits(output_det, dt_target, weight=wts, reduction='mean'))
            l2.append(F.l1_loss(output_votes, votes, reduction='mean'))

        l1 = torch.mean(torch.stack(l1))
        l2 = torch.mean(torch.stack(l2))
        t = l1 + l2

        #total_loss += l.item()
        det_loss += l1.item()
        scale_loss += l2.item()
        optimizer.zero_grad()
        t.backward()
        optimizer.step()

    print('Total Time for train epoch %.4f' % (time.time() - start_time))

    return det_loss, scale_loss


def train_test(data_loader, model):
    model.eval()
    #total_loss = 0
    start_time = time.time()
    det_loss = 0
    scale_loss = 0

    with torch.no_grad():
        for i, (img, det_maps, wts, votes, wvotes) in enumerate(tqdm(data_loader)):

            dt_target = det_maps.cuda(non_blocking=True)
            inputs = img.cuda(non_blocking=True)
            wts = wts.cuda(non_blocking=True)
            votes = votes.cuda(non_blocking=True)
            wvotes = wvotes.cuda(non_blocking=True)

            output = model(inputs)

            N = len(output)
            l1 = []
            l2 = []
            for n in range(N):
                output_det = output[n][:, 0:1]
                output_votes = output[n][:, 1:2] * wvotes
                l1.append(F.binary_cross_entropy_with_logits(output_det, dt_target, weight=wts, reduction='mean'))
                #l2.append(F.l1_loss(output_votes, votes, reduction='mean'))
                l2.append(F.smooth_l1_loss(output_votes, votes))

            l1 = torch.mean(torch.stack(l1))
            l2 = torch.mean(torch.stack(l2))
            #l = l1 + l2

        #total_loss += l.item()
        det_loss += l1.item()
        scale_loss += l2.item()

    print('Total Time for test epoch %.4f' % (time.time() - start_time))

    return det_loss, scale_loss


def main():
    args = parse_args()

    # __________________ Params ___________________
    sample_size = args.sample_size
    train_batch_size = args.train_batch_size
    test_batch_size = 128
    num_epochs = args.num_epochs
    num_workers = args.num_worker
    lr = args.learning_rate
    start_epoch = 0
    # _____________________________________________

    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.set_num_threads(num_workers + 1)
    cudnn.benchmark = True
    # cudnn.deterministic = False
    cudnn.enabled = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    coco_train = Coco('datasets/data/coco_data/person_keypoints_train2017.json', 'train', sample_size,
                      transforms.Compose([normalize]))
    coco_val = Coco('datasets/data/coco_data/person_keypoints_val2017.json', 'train', sample_size,
                    transforms.Compose([normalize]))

    train_loader = DataLoader(coco_train, batch_size=train_batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=False)
    val_loader = DataLoader(coco_val, batch_size=test_batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=False)

    pose_net = bninception(out_chn=2)
    model = DataParallel(pose_net)
    model.cuda()

    #checkpoint = torch.load('models/m_100.pth')
    #pretrained_dict = checkpoint['state_dict']
    #model.load_state_dict(pretrained_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #os.makedirs(args.log, exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        if epoch == 100:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4

        dloss, scale_loss = train(train_loader, model, optimizer)
        tloss = 'det_loss ' + str(dloss) + ' scale loss ' + str(scale_loss)

        dloss, scale_loss = train_test(val_loader, model)
        test_loss = 'det_loss ' + str(dloss) + ' scale loss ' + str(scale_loss)

        with open('losses/train_loss_384.txt', 'a') as the_file:
            the_file.write(str(tloss) + '\n')

        with open('losses/test_loss_384.txt', 'a') as the_file:
            the_file.write(str(test_loss) + '\n')

        ckpt = {
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict()
        }

        #os.makedirs(args.model_save_path, exist_ok=True)
        #ckpt_name = os.path.join(args.model_save_path, 'epoch_%d.ckpt' % epoch)
        torch.save(ckpt, 'models/m_384_' + str(epoch) +'.pth')


if __name__ == '__main__':
    main()
