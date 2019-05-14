import torch
import random
import time
import os
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel
from models.bn_inception import bninception
from data_utils import get_preds, apply_augmentation_test
import cv2

os.environ['CUDA_VISIBLE_DEVICES']='0'


manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
torch.set_num_threads(8)
cudnn.benchmark = True
cudnn.enabled = True

pose_net = bninception(out_chn=2)
model = DataParallel(pose_net)
model.cuda()
checkpoint = torch.load('models/m_129.pth')
pretrained_dict = checkpoint['state_dict']
model.load_state_dict(pretrained_dict)
model.eval()

start_time = time.time()
det_loss = 0
scale_loss = 0
rec = []
t = 0


# read image
image_path = './images/000015.jpg'
image = cv2.imread(image_path)
img, imagesf, warps = apply_augmentation_test(image_path, output_size=256)

with torch.no_grad():
    img = img.unsqueeze(dim=0)
    inputs = img.cuda()
    output = model(inputs)
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
    centre = det['center']
    x = int(centre[0])
    y = int(centre[1])
    scale = int(det['scale']/10)
    cv2.rectangle(image, (x-scale, y-scale), (x+scale, y+scale), (0, 255, 255), 2)

cv2.imshow("detections", image)
cv2.waitKey(0)

print(len(rec))
print(rec)

#
