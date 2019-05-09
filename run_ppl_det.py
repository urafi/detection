
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from torch.utils.data import DataLoader
from data_utils import *
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import random
import torch.backends.cudnn as cudnn
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from torch.nn import DataParallel
import torchvision
from bn_inception2 import bninception


flipRef = [i-1 for i in [1,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16]]

class Coco:

    def __init__(self, img_dir=None, dtype='train'):

        with open('data/coco_data/COCO_det.json') as anno_file:
            self.anno = json.load(anno_file)
        self.coco = COCO('/media/datasets/pose_estimation/MSCOCO_2017/annotations_trainval2017/annotations/person_keypoints_val2017.json')
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        #with open('data/coco_data/COCO_val2017_detections_AP_H_56_person.json') as anno_file:
        #     self.anno = json.load(anno_file)
        self.img_dir = '/media/datasets/pose_estimation/MSCOCO_2017/images/val2017/'

    def __len__(self):

        return len(self.anno)

    def __getitem__(self, ind):

        im = self.anno[ind]['image']#self.coco.loadImgs(self.anno[ind]['image_id'])[0]['file_name']
        path = im#self.img_dir + im
        images, imagesf, warps = apply_augmentation_test_td(path, self.anno[ind], output_size=256)
        meta = {'index': ind, 'imgID': self.anno[ind]['im_id'],
                'warps': warps, 'score': self.anno[ind]['score']}

        return {'images': images, 'imagesf': imagesf, 'meta': meta}


manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
cudnn.benchmark = True


coco_val = Coco(img_dir= '../Coco/val2017/', dtype='val')
test_batch_size = 128
val_loader = DataLoader(coco_val, batch_size=test_batch_size, shuffle=False, num_workers=8)


poseNet = bninception()
poseNet = poseNet.cuda()
model = DataParallel(poseNet)
checkpoint = torch.load('models/model_r222.pth')
pretrained_dict = checkpoint['state_dict']
model.load_state_dict(pretrained_dict)
model.eval()


det = []
image_ids = []
im_counter = 1
total_loss = 0
counter = 0
total = 0
truncated = 0
image_ids = []

for i_batch, sampled_batch in enumerate(val_loader):

    with torch.no_grad():

        images = sampled_batch['images']
        imagesf = sampled_batch['imagesf']
        inputs = Variable(images).cuda()
        inputsf = Variable(imagesf).cuda()



        output = model(inputs)
        output_det = F.sigmoid(output[1][:, 0:17, :, :])
        output_det = output_det.data.cpu()

        outputf = model(inputsf)
        output_detf = F.sigmoid(outputf[1][:, 0:17, :, :])
        output_detf = output_detf.data.cpu()

        sr = output[1][:, 17:51, :, :].data.cpu()

        print('Iter [%d/%d]' %(i_batch + 1, len(coco_val) // test_batch_size))

        N = output_det.shape[0]

        for n in range(N):

            single_result_dict = {}

            prs = torch.zeros(17, 64, 64)
            output_detf[n] = output_detf[n][flipRef]
            for j in range(17):
                prs[j] = output_det[n][j] + torch.from_numpy(cv2.flip(output_detf[n][j].numpy(), 1))

            keypoints, score = get_preds_td(prs, sampled_batch['meta']['warps'][n], sr[n])

            single_result_dict['image_id'] = int(sampled_batch['meta']['imgID'][n].item())
            single_result_dict['category_id'] = 1
            single_result_dict['keypoints'] = keypoints
            single_result_dict['score'] = score * sampled_batch['meta']['score'][n].item()
            image_ids.append(int(sampled_batch['meta']['imgID'][n].item()))

            det.append(single_result_dict)

            #OKS = compute_OKS(sampled_batch['meta']['kpts'][n], keypoints, sampled_batch['meta']['bbox'][n], sampled_batch['meta']['area'][n])
            #if OKS > .8:

            #    total += 1

            #    if sampled_batch['meta']['truncated'] == True:
            #        truncated += 1

            filename = 'output_images/im_' + str(im_counter) + '.jpg'
            #im = plot_masks_on_image(images[n], output_det[n])
            #torchvision.utils.save_image(im, filename)
            im_counter += 1


with open('dt.json', 'w') as f:
        json.dump(det, f)


# eval_gt = COCO('../Coco/annotations/person_keypoints_val2017.json')
eval_gt = COCO('/media/datasets/pose_estimation/MSCOCO_2017/annotations_trainval2017/annotations/person_keypoints_val2017.json')
eval_dt = eval_gt.loadRes('dt.json')
cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
#cocoEval.params.imgIds = list(image_ids)
#cocoEval.params.catIds = [1]
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()


print(total)
print(truncated)
