# python
import os
import argparse
import random
import numpy as np
import shutil

# pytorch
import torch

# 3rd-party utils
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# user-defined
from models.detr import DETR
import utils
from datagen import jsonDataset, ConcatBalancedDataset


class Tester(object):
    def __init__(self, config, trainset, validset, class_idx_map):
        self.config = config

        self.class_idx_map = class_idx_map

        '''cuda'''
        if torch.cuda.is_available() and not config['cuda']['using_cuda']:
            print("WARNING: You have a CUDA device, so you should probably run with using cuda")

        self.is_data_parallel = False
        if isinstance(config['cuda']['gpu_id'], list):
            self.is_data_parallel = True
            cuda_str = 'cuda:' + str(config['cuda']['gpu_id'][0])
        elif isinstance(config['cuda']['gpu_id'], int):
            cuda_str = 'cuda:' + str(config['cuda']['gpu_id'])
        else:
            raise ValueError('Check out gpu id in config')

        self.device = torch.device(cuda_str if config['cuda']['using_cuda'] else "cpu")

        '''Data'''
        assert trainset
        assert validset
        print('==> Preparing data..')
        ''' data loader'''
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=config['params']['batch_size'],
            shuffle=False, num_workers=config['params']['data_worker'],
            collate_fn=self.collate_fn,
            pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            validset, batch_size=config['params']['batch_size'],
            shuffle=False, num_workers=config['params']['data_worker'],
            collate_fn=self.collate_fn,
            pin_memory=True)

        self.dataloaders = {'train': train_loader, 'valid': valid_loader}

        '''tensorboard'''
        self.summary_writer = SummaryWriter(os.path.join(config['model']['exp_path'], 'log'))

    def collate_fn(self, batch):
        batch = list(zip(*batch))
        batch[0] = utils.nested_tensor_from_tensor_list(batch[0])
        return tuple(batch)

    def init_net(self, num_classes):
        '''Model'''
        self.net = DETR(config=self.config, num_classes=num_classes)
        self.net = self.net.to(self.device)

        best_ckpt_path = os.path.join(self.config['model']['exp_path'], 'best.pth')
        print(best_ckpt_path)
        ckpt = torch.load(best_ckpt_path, map_location=self.device)
        weights = utils._load_weights(ckpt['net'])
        missing_keys = self.net.load_state_dict(weights, strict=True)
        print(missing_keys)

        '''print out net'''
        print(self.net)
        n_parameters = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

    def print(self):
        utils.print_config(self.config)

        # input("Press any key to continue..")

    def test_on_data(self, dataset, img_size, out_dir):
        self.net.eval()
        dataloader = self.dataloaders[dataset]
        data_out_dir = os.path.join(out_dir, dataset)
        if not os.path.exists(data_out_dir):
            os.makedirs(data_out_dir, exist_ok=True)

        with torch.set_grad_enabled(False):
            for batch_idx, (inputs, targets, mask, paths) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                outputs = self.net(inputs)

                class_logits = outputs['pred_logits']
                bbox_preds = outputs['pred_boxes']

                utils._write_results(class_logits, bbox_preds, paths, self.class_idx_map, img_size, data_out_dir)


    def start(self, out_dir):
        img_size = self.config['params']['image_size'].split('x')
        img_size = (int(img_size[0]), int(img_size[1]))

        for dataset in self.dataloaders:
            self.test_on_data(dataset, img_size, out_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path of config file')
    opt = parser.parse_args()

    config = utils.get_config(opt.config)

    '''make output folder'''
    output_dir = os.path.join(config['model']['exp_path'], 'results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    '''set random seed'''
    random.seed(config['params']['random_seed'])
    np.random.seed(config['params']['random_seed'])
    torch.manual_seed(config['params']['random_seed'])

    '''Data'''
    target_classes = utils.read_txt(config['params']['classes'])
    num_classes = len(target_classes)
    class_idx_map = dict()
    # 0 is background class
    for idx in range(0, num_classes):
        class_idx_map[idx + 1] = target_classes[idx]

    img_size = config['params']['image_size'].split('x')
    img_size = (int(img_size[0]), int(img_size[1]))

    print('==> Preparing data..')
    bbox_params = A.BboxParams(format='pascal_voc', min_visibility=0.3)
    valid_transforms = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1], p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ], bbox_params=bbox_params, p=1.0)

    train_dataset = jsonDataset(path=config['data']['train'], classes=target_classes,
                                transforms=valid_transforms)

    valid_dataset = jsonDataset(path=config['data']['valid'], classes=target_classes,
                                transforms=valid_transforms)

    if 'add_train' in config['data'] and config['data']['add_train'] is not None:
        add_train_dataset = jsonDataset(path=config['data']['add_train'], classes=target_classes,
                                        transforms=valid_transforms)
        train_dataset = ConcatBalancedDataset([train_dataset, add_train_dataset])

    assert train_dataset
    assert valid_dataset

    tester = Tester(config=config, trainset=train_dataset, validset=valid_dataset, class_idx_map=class_idx_map)
    tester.init_net(num_classes=num_classes)
    tester.print()
    tester.start(output_dir)
