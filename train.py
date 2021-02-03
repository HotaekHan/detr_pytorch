# python
import os
import argparse
import random
import numpy as np
import shutil

# pytorch
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch import autograd

# 3rd-party utils
from torch.utils.tensorboard import SummaryWriter

# user-defined
from models.detr import DETR
from models.loss import SetCriterion
from models.box_matcher import HungarianMatcher
import utils



class Trainer(object):
    def __init__(self, config):
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

        '''tensorboard'''
        self.summary_writer = SummaryWriter(os.path.join(config['model']['exp_path'], 'log'))

    def init_net(self):
        '''Model'''
        num_classes = len(target_classes)

        self.net = DETR(config=self.config)
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
                                   cost_giou=config['boxt_matcher']['cost_giou'])
        weight_dict = {'loss_ce': 1, 'loss_bbox': config['params']['bbox_loss_coef']}
        weight_dict['loss_giou'] = config['params']['giou_loss_coef']
        # if args.masks:
        #     weight_dict["loss_mask"] = args.mask_loss_coef
        #     weight_dict["loss_dice"] = args.dice_loss_coef
        # TODO this is a hack
        if config['detr']['aux_loss']:
            aux_weight_dict = {}
            for i in range(config['transformer']['dec_layers'] - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ['labels', 'boxes', 'cardinality']
        # if args.masks:
        #     losses += ["masks"]
        self.criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                                      eos_coef=config['params']['eos_coef'], losses=losses)
        self.criterion.to(self.device)

        '''optimizer'''
        param_dicts = [
            {"params": [p for n, p in self.net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": config['basenet']['lr_backbone'],
            },
        ]
        if config['params']['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=float(config['params']['lr']), momentum=0.9,
                                       weight_decay=5e-4)
        elif config['params']['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=float(config['params']['lr']))
        elif config['params']['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                          weight_decay=args.weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
        else:
            raise ValueError('not supported optimizer')

    def print(self):
        # print out
        print("optimizer : " + str(self.optimizer))
        print("Size of batch : " + str(self.train_loader.batch_size))
        print("transform : " + str(self.transform))


        print("num. train data : " + str(len(train_dataset)))
        print("num. valid data : " + str(len(valid_dataset)))
        print("num_classes : " + str(self.num_classes))
        print("classes : " + str(target_classes))

        utils.print_config(config)

        input("Press any key to continue..")

    def iterate(self, epoch, phase):
        is_train = False
        if phase == 'train':
            self.net.train()
            is_train = True
        else:
            self.net.eval()
        acc_loss = 0.
        avg_loss = 0.

        with torch.set_grad_enabled(is_train):
            for batch_idx, (inputs, loc_targets, cls_targets, mask_targets, paths) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                loc_targets = loc_targets.to(self.device)
                cls_targets = cls_targets.to(self.device)

                loc_preds, cls_preds = self.net(inputs)

                loc_loss, cls_loss, mask_loss, num_matched_anchors = \
                    criterion(loc_preds, loc_targets, cls_preds, cls_targets, mask_preds, mask_targets)
                if num_matched_anchors == 0:
                    print('No matched anchor')
                    continue
                else:
                    num_matched_anchors = float(num_matched_anchors)
                    loss = ((loc_loss + cls_loss) / num_matched_anchors) + mask_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc_loss += loss.item()
                avg_loss = acc_loss / (batch_idx + 1)
                print('[%s] epoch: %3d | iter: %4d | loc_loss: %.3f | cls_loss: %.3f | '
                      'train_loss: %.3f | avg_loss: %.3f | matched_anchors: %d'
                      % (phase, epoch, batch_idx, loc_loss.item(), cls_loss.item(), loss.item(),
                         avg_loss, num_matched_anchors))

                if is_train is True:
                    self.summary_writer.add_scalar('train/loc_loss', loc_loss.item(), self.global_iter_train)
                    self.summary_writer.add_scalar('train/cls_loss', cls_loss.item(), self.global_iter_train)
                    self.summary_writer.add_scalar('train/mask_loss', mask_loss.item(), self.global_iter_train)
                    self.summary_writer.add_scalar('train/train_loss', loss.item(), self.global_iter_train)
                    self.global_iter_train += 1
                else:
                    self.summary_writer.add_scalar('valid/loc_loss', loc_loss.item(), self.global_iter_valid)
                    self.summary_writer.add_scalar('valid/cls_loss', cls_loss.item(), self.global_iter_valid)
                    self.summary_writer.add_scalar('valid/mask_loss', mask_loss.item(), self.global_iter_valid)
                    self.summary_writer.add_scalar('valid/train_loss', loss.item(), self.global_iter_valid)
                    self.global_iter_valid += 1

        return avg_loss

    def train(self):
        for epoch in range(self.start_epoch, self.config['params']['epoch'], 1):
            self.iterate(epoch, "train")

            state = {
                "epoch": epoch,
                "best_loss": self.best_valid_loss,
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            torch.save(state, os.path.join(self.config['exp']['path'], 'latest.pth'))

            valid_loss = self.iterate(epoch, "valid")
            if valid_loss < self.best_valid_loss:
                print("******** New optimal found, saving state ********")
                self.best_valid_loss = valid_loss
                # torch.save(state, os.path.join(self.exp_dir, "ckpt-" + str(epoch) + '.pth'))
                torch.save(state, os.path.join(self.config['exp']['path'], 'best.pth'))
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
        os.mkdir(config['model']['exp_path'])

    if not os.path.exists(os.path.join(config['model']['exp_path'], 'config.yaml')):
        shutil.copy(opt.config, os.path.join(config['model']['exp_path'], 'config.yaml'))
    else:
        os.remove(os.path.join(config['model']['exp_path'], 'config.yaml'))
        shutil.copy(opt.config, os.path.join(config['model']['exp_path'], 'config.yaml'))

    '''set random seed'''
    random.seed(config['hyperparameters']['random_seed'])
    np.random.seed(config['hyperparameters']['random_seed'])
    torch.manual_seed(config['hyperparameters']['random_seed'])

    # Data
    print('==> Preparing data..')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    target_classes = config['hyperparameters']['classes'].split('|')
    img_size = config['hyperparameters']['image_size'].split('x')
    img_size = (int(img_size[0]), int(img_size[1]))

    train_dataset = jsonDataset(path=config['data']['train'].split(' ')[0], classes=target_classes,
                                transform=transform,
                                input_image_size=img_size,
                                num_crops=config['hyperparameters']['num_crops'],
                                do_aug=config['hyperparameters']['do_aug'])

    valid_dataset = jsonDataset(path=config['data']['valid'].split(' ')[0], classes=target_classes,
                                transform=transform,
                                input_image_size=img_size,
                                num_crops=config['hyperparameters']['num_crops'])

    assert train_dataset
    assert valid_dataset

    if config['data']['add_train'] is not None:
        add_train_dataset = jsonDataset(path=config['data']['add_train'].split(' ')[0], classes=target_classes,
                                transform=transform,
                                input_image_size=img_size,
                                num_crops=config['hyperparameters']['num_crops'],
                                do_aug=config['hyperparameters']['do_aug'])
        concat_train_dataset = ConcatBalancedDataset([train_dataset, add_train_dataset])
        assert add_train_dataset
        assert concat_train_dataset

        train_loader = torch.utils.data.DataLoader(
            concat_train_dataset, batch_size=config['hyperparameters']['batch_size'],
            shuffle=True, num_workers=config['hyperparameters']['data_worker'],
            collate_fn=train_dataset.collate_fn,
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['hyperparameters']['batch_size'],
            shuffle=True, num_workers=config['hyperparameters']['data_worker'],
            collate_fn=train_dataset.collate_fn,
            pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config['hyperparameters']['batch_size'],
        shuffle=False, num_workers=config['hyperparameters']['data_worker'],
        collate_fn=valid_dataset.collate_fn,
        pin_memory=True)



