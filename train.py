# python
import os
import argparse
import random
import numpy as np
import shutil

# pytorch
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import autograd

# 3rd-party utils
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# user-defined
from models.detr import DETR
from models.loss import SetCriterion
from models.box_matcher import HungarianMatcher
import utils
from datagen import jsonDataset, ConcatBalancedDataset


class Trainer(object):
    def __init__(self, config, trainset, validset):
        self.config = config
        '''variables'''
        self.best_valid_loss = float('inf')
        self.global_iter_train = 0
        self.global_iter_valid = 0
        self.start_epoch = 0

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
            shuffle=True, num_workers=config['params']['data_worker'],
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

        '''print out net'''
        print(self.net)
        n_parameters = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        '''set data parallel'''
        if self.is_data_parallel is True:
            self.net = torch.nn.DataParallel(module=self.net, device_ids=config['cuda']['gpu_id'])

        '''loss'''
        matcher = HungarianMatcher(cost_class=config['box_matcher']['cost_class'],
                                   cost_bbox=config['box_matcher']['cost_bbox'],
                                   cost_giou=config['box_matcher']['cost_giou'])
        loss_weight = dict()
        loss_weight['loss_ce'] = 1.0
        loss_weight['loss_bbox'] = float(config['loss']['bbox_loss_coef'])
        loss_weight['loss_giou'] = float(config['loss']['giou_loss_coef'])
        # if args.masks:
        #     weight_dict["loss_mask"] = args.mask_loss_coef
        #     weight_dict["loss_dice"] = args.dice_loss_coef
        # TODO this is a hack
        if config['detr']['aux_loss']:
            aux_weight_dict = dict()
            for i in range(config['transformer']['dec_layers'] - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in loss_weight.items()})
            loss_weight.update(aux_weight_dict)
        losses = ['labels', 'boxes', 'cardinality']
        # if args.masks:
        #     losses += ["masks"]
        self.criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=loss_weight,
                                      eos_coef=config['loss']['eos_coef'], losses=losses)
        self.criterion.to(self.device)

        '''optimizer'''
        param_dicts = [
            {"params": [p for n, p in self.net.named_parameters() if "basenet" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.net.named_parameters() if "basenet" in n and p.requires_grad],
                "lr": float(config['basenet']['lr_backbone']),
            },
        ]
        if config['params']['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(param_dicts, lr=float(config['params']['lr']), momentum=0.9,
                                       weight_decay=5e-4)
        elif config['params']['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(param_dicts, lr=float(config['params']['lr']))
        elif config['params']['optimizer'] == 'AdamW':
            self.optimizer = optim.AdamW(param_dicts, lr=float(self.config['params']['lr']),
                                         weight_decay=float(self.config['params']['weight_decay']))
            # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
        else:
            raise ValueError('not supported optimizer')

    def print(self):
        # print out
        print("optimizer : " + str(self.optimizer))
        print("Size of batch : " + str(self.dataloaders['train'].batch_size))
        # print("transform : " + str(self.transform_train))
        # print("num. train data : " + str(len(train_dataset)))
        # print("num. valid data : " + str(len(valid_dataset)))
        # print("num_classes : " + str(self.num_classes))
        # print("classes : " + str(target_classes))

        utils.print_config(config)

        # input("Press any key to continue..")

    def train_one_epoch(self, epoch, phase):
        is_train = False
        if phase == 'train':
            self.net.train()
            self.criterion.train()
            is_train = True
        else:
            self.net.eval()
            self.criterion.eval()
        acc_loss = 0.
        avg_loss = 0.

        dataloader = self.dataloaders[phase]

        with torch.set_grad_enabled(is_train):
            for batch_idx, (inputs, targets, mask, paths) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                outputs = self.net(inputs)

                loss_dict = self.criterion(outputs, targets)
                weight_dict = self.criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                if is_train is True:
                    self.optimizer.zero_grad()
                    losses.backward()
                    self.optimizer.step()

                acc_loss += losses.item()
                avg_loss = acc_loss / (batch_idx + 1)
                print(f'[{phase}] epoch: {epoch:3d} | iter: {batch_idx:4d} | avg_loss: {avg_loss:.4f} | '
                      f'ce_loss: {loss_dict["loss_ce"]:.4f} | bbox_loss: {loss_dict["loss_bbox"]:.4f} | '
                      f'giou_loss: {loss_dict["loss_giou"]:.4f}')

                if is_train is True:
                    self.summary_writer.add_scalar('train/avg_loss', avg_loss, self.global_iter_train)
                    self.summary_writer.add_scalar('train/ce_loss', loss_dict['loss_ce'], self.global_iter_train)
                    self.summary_writer.add_scalar('train/bbox_loss', loss_dict['loss_bbox'], self.global_iter_train)
                    self.summary_writer.add_scalar('train/giou_loss', loss_dict['loss_giou'], self.global_iter_train)
                    self.global_iter_train += 1
                else:
                    self.summary_writer.add_scalar('valid/avg_loss', avg_loss, self.global_iter_valid)
                    self.summary_writer.add_scalar('valid/ce_loss', loss_dict['loss_ce'], self.global_iter_valid)
                    self.summary_writer.add_scalar('valid/bbox_loss', loss_dict['loss_bbox'], self.global_iter_valid)
                    self.summary_writer.add_scalar('valid/giou_loss', loss_dict['loss_giou'], self.global_iter_valid)
                    self.global_iter_valid += 1

        return avg_loss

    def start(self):
        for epoch in range(self.start_epoch, self.config['params']['epoch'], 1):
            self.train_one_epoch(epoch, "train")

            state = {
                "epoch": epoch,
                "best_loss": self.best_valid_loss,
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            torch.save(state, os.path.join(self.config['model']['exp_path'], 'latest.pth'))

            valid_loss = self.train_one_epoch(epoch, "valid")
            if valid_loss < self.best_valid_loss:
                print("******** New optimal found, saving state ********")
                self.best_valid_loss = valid_loss
                # torch.save(state, os.path.join(self.exp_dir, "ckpt-" + str(epoch) + '.pth'))
                torch.save(state, os.path.join(self.config['model']['exp_path'], 'best.pth'))
            print()
        self.summary_writer.close()
        print("best valid loss : " + str(self.best_valid_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path of config file')
    opt = parser.parse_args()

    config = utils.get_config(opt.config)

    '''make output folder'''
    if not os.path.exists(config['model']['exp_path']):
        os.makedirs(config['model']['exp_path'], exist_ok=True)

    if not os.path.exists(os.path.join(config['model']['exp_path'], 'config.yaml')):
        shutil.copy(opt.config, os.path.join(config['model']['exp_path'], 'config.yaml'))
    else:
        os.remove(os.path.join(config['model']['exp_path'], 'config.yaml'))
        shutil.copy(opt.config, os.path.join(config['model']['exp_path'], 'config.yaml'))

    '''set random seed'''
    random.seed(config['params']['random_seed'])
    np.random.seed(config['params']['random_seed'])
    torch.manual_seed(config['params']['random_seed'])

    '''Data'''
    target_classes = utils.read_txt(config['params']['classes'])
    num_classes = len(target_classes)
    img_size = config['params']['image_size'].split('x')
    img_size = (int(img_size[0]), int(img_size[1]))

    print('==> Preparing data..')
    bbox_params = A.BboxParams(format='pascal_voc', min_visibility=0.3)
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.Sequential([
                A.Resize(height=img_size[0], width=img_size[1], p=1.0),
            ], p=1.0),
            A.Sequential([
                A.RandomSizedBBoxSafeCrop(height=img_size[0], width=img_size[1], p=1.0),
            ], p=1.0)
        ], p=1.0),

        A.OneOf([
            A.Sequential([
                A.GaussNoise(var_limit=(100, 150), p=0.5),
                A.MotionBlur(blur_limit=17, p=0.5)
            ], p=1.0),
            A.Sequential([
                A.GaussNoise(var_limit=(100, 150), p=0.5),
                A.MotionBlur(blur_limit=17, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=0, p=0.5),
            ], p=1.0),
            A.Sequential([
                A.GaussNoise(var_limit=(100, 150), p=0.5),
                A.MotionBlur(blur_limit=17, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=0, p=0.5),
                A.ChannelShuffle(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=(-0.15, 0.15), rotate_limit=30, p=0.5,
                                   border_mode=cv2.BORDER_CONSTANT, value=0),
            ], p=1.0)
        ], p=1.0),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ], bbox_params=bbox_params, p=1.0)
    valid_transforms = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1], p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ], bbox_params=bbox_params, p=1.0)

    train_dataset = jsonDataset(path=config['data']['train'], classes=target_classes,
                                transforms=train_transforms)

    valid_dataset = jsonDataset(path=config['data']['valid'], classes=target_classes,
                                transforms=valid_transforms)

    if 'add_train' in config['data'] and config['data']['add_train'] is not None:
        add_train_dataset = jsonDataset(path=config['data']['add_train'], classes=target_classes,
                                        transforms=train_transforms)
        train_dataset = ConcatBalancedDataset([train_dataset, add_train_dataset])

    assert train_dataset
    assert valid_dataset

    trainer = Trainer(config=config, trainset=train_dataset, validset=valid_dataset)
    trainer.init_net(num_classes=num_classes)
    trainer.print()
    trainer.start()
