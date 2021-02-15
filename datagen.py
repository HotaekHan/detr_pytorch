'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg width height xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
import random
import numpy as np
import json
import os
import cv2

import torch
import torch.utils.data as data


class jsonDataset(data.Dataset):
    def __init__(self, path, classes, transforms, view_image=False,
                 min_cols=1, min_rows=1):

        self.path = path
        self.classes = classes
        self.transforms = transforms
        self.view_img = view_image

        self.fnames = list()
        self.boxes = list()
        self.labels = list()

        self.num_classes = len(self.classes)

        self.label_map = dict()
        self.class_idx_map = dict()
        # 0 is background class
        for idx in range(0, self.num_classes):
            self.label_map[self.classes[idx]] = idx+1
            self.class_idx_map[idx+1] = self.classes[idx]

        fp_read = open(self.path, 'r')
        gt_dict = json.load(fp_read)

        all_boxes = list()
        all_labels = list()
        all_img_path = list()

        '''read gt files'''
        for gt_key in gt_dict:
            gt_data = gt_dict[gt_key][0]

            box = list()
            label = list()

            num_boxes = len(gt_data['labels'])

            img = cv2.imread(gt_data['image_path'])
            img_rows = img.shape[0]
            img_cols = img.shape[1]

            for iter_box in range(0, num_boxes):
                xmin = gt_data['boxes'][iter_box][0]
                ymin = gt_data['boxes'][iter_box][1]
                xmax = gt_data['boxes'][iter_box][2]
                ymax = gt_data['boxes'][iter_box][3]
                rows = ymax - ymin
                cols = xmax - xmin

                if xmin < 0 or ymin < 0:
                    if xmin == -1:
                        xmin = 0
                    elif ymin == -1:
                        ymin = 0
                    else:
                        print('negative coordinate: [xmin: ' + str(xmin) + ', ymin: ' + str(ymin) + ']')
                        print(gt_data['image_path'])

                if xmax >= img_cols or ymax >= img_rows:
                    if xmax == img_cols:
                        xmax = img_cols - 1
                    elif ymax == img_rows:
                        ymax = img_rows -1
                    else:
                        print('over maximum size: [xmax: ' + str(xmax) + ', ymax: ' + str(ymax) + ']')
                        print(gt_data['image_path'])

                if cols < min_cols:
                    print('cols is lower than ' + str(min_cols) + ': [' + str(xmin) + ', ' + str(ymin) + ', ' +
                          str(xmax) + ', ' + str(ymax) + '] '
                          + str(gt_data['image_path']))
                    continue
                if rows < min_rows:
                    print('rows is lower than ' + str(min_rows) + ': [' + str(xmin) + ', ' + str(ymin) + ', ' +
                          str(xmax) + ', ' + str(ymax) + '] '
                          + str(gt_data['image_path']))
                    continue

                class_name = gt_data['labels'][iter_box][0]
                if class_name not in self.label_map:
                    print('weired class name: ' + class_name)
                    print(gt_data['image_path'])
                    continue

                class_idx = self.label_map[class_name]
                box.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                label.append(int(class_idx))

            # if len(box) == 0 or len(label) == 0:
            #     print('none of object exist in the image: ' + gt_data['image_path'])
            #     continue

            all_boxes.append(box)
            all_labels.append(label)
            all_img_path.append(gt_data['image_path'])

        if len(all_boxes) == len(all_labels) and len(all_boxes) == len(all_img_path):
            num_images = len(all_img_path)
        else:
            print('num. of boxes: ' + str(len(all_boxes)))
            print('num. of labels: ' + str(len(all_labels)))
            print('num. of paths: ' + str(len(all_img_path)))
            raise ValueError('num. of elements are different(all boxes, all_labels, all_img_path)')

        for idx in range(0, num_images, 1):
            self.fnames.append(all_img_path[idx])
            self.boxes.append(torch.tensor(all_boxes[idx], dtype=torch.float32))
            self.labels.append(torch.tensor(all_labels[idx], dtype=torch.int64))

        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        '''Load image and boxes.'''
        fname = self.fnames[idx]
        boxes = self.boxes[idx]
        labels = self.labels[idx]
        img = cv2.imread(fname)

        bboxes = [bbox.tolist() + [label.item()] for bbox, label in zip(boxes, labels)]
        augmented = self.transforms(image=img, bboxes=bboxes)
        # img = np.ascontiguousarray(augmented['image'])
        img = augmented['image']
        boxes = augmented['bboxes']
        boxes = [list(bbox) for bbox in boxes]
        labels = [bbox.pop() for bbox in boxes]
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        mask = torch.zeros(img.shape[0], img.shape[1], dtype=torch.int64)

        for box in boxes:
            mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1

        if self.view_img is True:
            for idx_box, box in enumerate(boxes):
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0))
                class_idx = labels[idx_box].item()
                text_size = cv2.getTextSize(self.class_idx_map[class_idx], cv2.FONT_HERSHEY_PLAIN, 1, 1)
                cv2.putText(img, self.class_idx_map[class_idx], (int(box[0]), int(box[1]) - text_size[1]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            cv2.imwrite(os.path.join("crop_test", str(idx)+".jpg"), img)
            mask_img = np.uint8(mask * 255.)
            cv2.imwrite(os.path.join("crop_test", str(idx)+"_mask.jpg"), mask_img)

        target = dict()
        target['boxes'] = boxes
        target['labels'] = labels

        return img, target, mask, fname

    def __len__(self):
        return self.num_samples

    # def _resize(self, img, boxes):
    #     if isinstance(self.input_size, int) is True:
    #         w = h = self.input_size
    #     elif isinstance(self.input_size, tuple) is True:
    #         h = self.input_size[0]
    #         w = self.input_size[1]
    #     else:
    #         raise ValueError('input size should be int or tuple of ints')
    #
    #     ws = 1.0 * w / img.shape[1]
    #     hs = 1.0 * h / img.shape[0]
    #     scale = torch.tensor([ws, hs, ws, hs], dtype=torch.float32)
    #     if boxes.numel() == 0:
    #         scaled_box = boxes
    #     else:
    #         scaled_box = scale * boxes
    #     return cv2.resize(img, (w, h)), scaled_box


class ConcatBalancedDataset(data.Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """
    @staticmethod
    def get_sizes(sequence):
        out_list = list()
        for seq in sequence:
            out_list.append(len(seq))
        return out_list

    def __init__(self, datasets):
        super(ConcatBalancedDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.dataset_sizes = self.get_sizes(self.datasets)
        self.num_datasets = len(self.datasets)

    def __len__(self):
        return min(self.dataset_sizes)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = int(idx % self.num_datasets)
        sample_idx = int(idx / self.num_datasets)
        sample_idx = sample_idx % self.dataset_sizes[dataset_idx]
        return self.datasets[dataset_idx][sample_idx]


def test():
    import torchvision

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    # ])
    # set random seed
    random.seed(777)
    np.random.seed(777)
    torch.manual_seed(777)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    classes = 'aeroplane|bicycle|bird|boat|bottle|bus|car|cat|chair|cow|diningtable|dog|horse|motorbike|person|pottedplant|sheep|sofa|train|tvmonitor'
    classes = classes.split('|')

    dataset1 = jsonDataset(path='data/voc.json', classes=classes, transform=transform,
                          input_image_size=(300, 300), num_crops=1, view_image=True, do_aug=True)
    print(len(dataset1))
    dataset2 = jsonDataset(path='data/cifar.json', classes=classes, transform=transform,
                          input_image_size=(150, 150), num_crops=1, view_image=True, do_aug=True)
    print(len(dataset2))

    dataset = ConcatBalancedDataset([dataset1, dataset2])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=False, num_workers=0,
                                             collate_fn=dataset1.collate_fn)

    while True:
        for idx, (images, loc_targets, cls_targets, mask_targets, paths) in enumerate(dataloader):
            np_img = images.numpy()
            print(images.size())
            print(loc_targets.size())
            print(cls_targets.size())
            print(mask_targets.size())
        break

if __name__ == '__main__':
    test()
