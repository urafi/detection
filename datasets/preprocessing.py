"""
 * preprocessing
 * Created on 03.04.19
 * Author: doering
"""
import json
import numpy as np
import os
import argparse
from pycocotools.coco import COCO
from tqdm import tqdm


def preprocess_coco(image_path, anno_path):

    coco = COCO(anno_path)
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)
    images = coco.loadImgs(img_ids)
    #num_joints = 17

    img_annos = []
    for idx, im in enumerate(tqdm(images)):
        img = os.path.join(image_path, im['file_name'])
        ann_id = coco.getAnnIds(imgIds=im['id'], catIds=cat_ids)

        anns = coco.loadAnns(ann_id)
        width = im['width']
        height = im['height']

        valid_anns = []
        rec = []
        for id, ann in enumerate(anns):

            x, y, w, h = ann['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if ann['area'] > 0 and x2 >= x1 and y2 >= y1:
                ann['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_anns.append(ann)

        for ann in valid_anns:

            if ann['iscrowd']:
                rec.append({
                    'image': img,
                    'iscrowd': ann['iscrowd'],
                    'im_id': im['id'],
                    'mask': ann['segmentation']
                })
                continue

            X = np.array(ann['keypoints'][0:51:3])
            Y = np.array(ann['keypoints'][1:51:3])
            state = ann['keypoints'][2:51:3]
            #
            if len(X[X > 0]) == 0 or len(Y[Y > 0]) == 0:
                continue

            if len(X[X > 0]) == 1 or len(Y[Y > 0]) == 1:
                continue

            scale = float(np.maximum(ann['clean_bbox'][2], ann['clean_bbox'][3]))
            center = [float(ann['clean_bbox'][0] + (ann['clean_bbox'][2] / 2)),
                      float(ann['clean_bbox'][1] + (ann['clean_bbox'][3] / 2))]

            rec.append({
                'image': img,
                'center': center,
                'scale': scale,
                'im_id': im['id'],
                'area': ann['area'],
                'keypoints': ann['keypoints'],
                'bbox': ann['bbox'],
                'iscrowd' : ann['iscrowd'],
                'mask' : ann['segmentation']
            })

        if len(rec) > 0:
            img_annos.append(rec)

    print(len(img_annos))
    os.makedirs('data/coco_data/', exist_ok=True)
    filename = os.path.splitext(anno_path.split('/')[-1])[0]
    with open('data/coco_data/%s.json' % filename, 'w') as f:
        json.dump(img_annos, f)


def preprocess_posetrack(data_path, anno_path, prefix):

    _labeled_only_ = prefix == 'train'

    posetrack_annotations_fp = os.path.join(anno_path, '%s' % prefix)

    anno_files = os.listdir('../PoseTrack/posetrack_data/annotations/%s/' % prefix)

    empty_rec = []
    
    for idx, file in tqdm(enumerate(anno_files)):
        api = COCO(os.path.join(posetrack_annotations_fp, file))
        img_ids = api.getImgIds()
        imgs = api.loadImgs(img_ids)

        for img_idx, img in enumerate(imgs):

            if not img['is_labeled'] and _labeled_only_:
                # we need to create a file with non annotated frames since evaluation toolkit sucks
                empty_rec.append({
                    'image': img['file_name'],

                    'img': img['frame_id'],
                    'track_id': -1,
                    'file_id': file,
                    'vid_id': img['vid_id'],
                    'seq_name': img['file_name'].split('/')[-2],
                    'frame_idx': img_idx,
                    'tot_frames': len(imgs)
                })
                continue

            ann_ids = api.getAnnIds(imgIds=img['id'])
            anns = api.loadAnns(ann_ids)

            added_anns = 0
    
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', type=str, default='coco')
    parser.add_argument('-dp', '--data_path', type=str,
                        default='/media/datasets/pose_estimation/MSCOCO_2017/images/val2017')
    parser.add_argument('-ap', '--anno_path', type=str,
                        default='/media/datasets/pose_estimation/MSCOCO_2017/annotations_trainval2017/annotations/person_keypoints_val2017.json')
    parser.add_argument('-pr', '--prefix', type=str, default='train')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    if args.dataset == 'coco':
        preprocess_coco(image_path=args.data_path,
                        anno_path=args.anno_path)
    elif args.dataset == 'posetrack':
        preprocess_posetrack(args.data_path, args.anno_path, args.prefix)

