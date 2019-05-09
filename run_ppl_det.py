import torch
import random
import time
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from datasets.coco import Coco
from argparser import parse_args
from torch.nn import DataParallel
from models.bn_inception import bninception
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_utils import get_preds, apply_augmentation_test
import json
from pycocotools.coco import COCO

args = parse_args()

# __________________ Params ___________________

test_batch_size = 32

# _____________________________________________


class Coco:

    def __init__(self):

        self.coco = COCO('/media/datasets/pose_estimation/MSCOCO_2017/annotations_trainval2017/annotations/person_keypoints_val2017.json')
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        img_ids = self.coco.getImgIds(catIds=self.cat_ids)
        self.anno = self.coco.loadImgs(img_ids)
        self.img_dir = '/media/datasets/pose_estimation/MSCOCO_2017/images/val2017/'

    def __len__(self):

        return len(self.anno)

    def __getitem__(self, ind):

        ann_id = self.coco.getAnnIds(imgIds=self.anno[ind]['id'], catIds=self.cat_ids)
        anns = self.coco.loadAnns(ann_id)
        path = self.img_dir + self.anno[ind]['file_name']
        images, imagesf, warps = apply_augmentation_test(path, output_size=256)
        N = len(anns)
        meta = {'img': path, 'imgID': self.anno[ind]['id'],
                'warps': warps, 'L': N}

        return images, imagesf, meta

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
torch.set_num_threads(8)
cudnn.benchmark = True
    # cudnn.deterministic = False
cudnn.enabled = True

coco_val = Coco()

val_loader = DataLoader(coco_val, batch_size=test_batch_size, shuffle=False, num_workers=8,
                            pin_memory=False)

pose_net = bninception(out_chn=2)
model = DataParallel(pose_net)
model.cuda()
checkpoint = torch.load('models/m_129.pth')
pretrained_dict = checkpoint['state_dict']
model.load_state_dict(pretrained_dict)
model.eval()
# total_loss = 0
start_time = time.time()
det_loss = 0
scale_loss = 0
rec = []
t = 0
for _, (img, imgf, meta) in enumerate(tqdm(val_loader)):
    with torch.no_grad():


        inputs = img.cuda(non_blocking=True)
        output = model(inputs)
        output_det = torch.sigmoid(output[1][:, 0:1]).data.cpu()
        output_votes = output[1][:, 1:2].data.cpu()
        N = output_det.shape[0]
        dets = []

        for i in range(N):
            dt, sc = get_preds(output_det[i], output_votes[i], meta['warps'][i])
            Ndets = len(sc)
            t += meta['L'][i].item()
            for j in range(Ndets):

                rec.append({#'image': meta['img'][i],
                            'center': [dt[0, j].item(), dt[1, j].item()],
                            'scale': sc[j].item(),
                            'im_id': meta['imgID'][0].item()})

print(len(rec))
print(t)
with open('data/coco_data/COCO_det.json', 'w') as f:
        json.dump(rec, f)

