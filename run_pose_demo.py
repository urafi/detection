import torch
import random
import time
import os
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel
from models.bn_inception import bninception
from bn_inception2 import bninception as pose
from data_utils import get_preds, apply_augmentation_test, apply_augmentation_test_td, get_preds_td
import cv2
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES']='0'


manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
torch.set_num_threads(8)
cudnn.benchmark = True
cudnn.enabled = True

pose_net = bninception(out_chn=2)
model_det = DataParallel(pose_net)
model_det.cuda()
checkpoint = torch.load('models/m_512_59.pth')
pretrained_dict = checkpoint['state_dict']
model_det.load_state_dict(pretrained_dict)
model_det.eval()

poseNet = pose()
poseNet = poseNet.cuda()
model_pose = DataParallel(poseNet)
checkpoint = torch.load('models/model_r222.pth')
pretrained_dict = checkpoint['state_dict']
model_pose.load_state_dict(pretrained_dict)
model_pose.eval()

start_time = time.time()
det_loss = 0
scale_loss = 0
rec = []
t = 0


# read image
image_path = 'data/000015.jpg'
image = cv2.imread(image_path)
img, imagesf, warps = apply_augmentation_test(image_path, output_size=512)

with torch.no_grad():
    img = img.unsqueeze(dim=0)
    inputs = img.cuda()
    output = model_det(inputs)
    output_det = torch.sigmoid(output[1][:, 0:1]).data.cpu()
    output_votes = output[1][:, 1:2].data.cpu()
    N = output_det.shape[0]
    dets = []
    for i in range(N):
        dt, sc, scores = get_preds(output_det, output_votes[i], warps.float())
        Ndets = len(sc)
        for j in range(Ndets):
            rec.append({
                        'center': [dt[0, j].item(), dt[1, j].item()],
                        'scale': sc[j].item(),

                        'score': scores[j].item()})

# visualise it
for det in rec:

    example = {'center': det['center'], 'scale' : det['scale']}
    im, imf, warp = apply_augmentation_test_td(image_path, example, 256)
    with torch.no_grad():
        im = im.unsqueeze(dim=0)
        inputs = im.cuda()
        output = model_pose(inputs)
        output_det = torch.sigmoid(output[1][:, 0:17, :, :]).data.cpu()
        sr = output[1][:, 17:51, :, :].data.cpu()
        keypoints, score = get_preds_td(output_det, warp, sr[0])
        X = keypoints[0:-1:3]
        Y = keypoints[1:-1:3]
        x1, x2 = int(min(X))-5, int(max(X))+5
        y1, y2 = int(min(Y))-5, int(max(Y))+5
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

cv2.imshow("detections", image)
cv2.waitKey(0)



