from pycocotools.coco import COCO
import cv2
import os
import json
from tqdm import tqdm

is_out = True
out_path = 'coco_trainval35k.json'
# out_path = 'coco_minival5k.json'

# annFile = '/coco/annotations/instances_valminusminival2014.json'
annFile = '/data/Projects/5_object_detection/coco/annotations/instances_valminusminival2014.json'
# annFile = '/data/Projects/5_object_detection/coco/annotations/instances_minival2014.json'
coco_root = '/data/Projects/5_object_detection/coco'

coco = COCO(annFile)
categories = coco.loadCats(coco.getCatIds())
class_names = [cat['name'] for cat in categories]
print('COCO categories: \n{}\n'.format(' '.join(class_names)))

img_ids = coco.getImgIds()

out_dict = dict()
img_idx = 0
for img_id in tqdm(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    img_name = img_info['file_name']
    dir_name = img_name.split('_')[1]
    img_path = os.path.join(coco_root, dir_name, img_name)
    img = cv2.imread(img_path)
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    boxes = list()
    labels = list()
    for ann in anns:
        bbox = ann['bbox']
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = xmin + bbox[2]
        ymax = ymin + bbox[3]
        class_name = coco.loadCats(ann['category_id'])[0]['name']

        # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0))
        # cv2.putText(img, class_name, (xmin, ymin), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append([class_name])

    gt_key = 'img_' + str(img_idx)
    img_idx += 1
    out_dict[gt_key] = list()
    out_data = dict()
    out_data['boxes'] = boxes
    out_data['labels'] = labels
    out_data['image_path'] = img_path

    out_dict[gt_key].append(out_data)

if is_out is True:
    json_str = json.dumps(out_dict, indent=4)
    out_file = open(out_path, 'w')
    out_file.writelines(json_str)
    out_file.close()
