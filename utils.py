from typing import Optional, List
import os
import collections

import torch
from torch import Tensor

import yaml
import cv2

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
if float(torchvision.__version__[:3]) < 0.7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size

def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # if torchvision._is_tracing():
        #     # nested_tensor_from_tensor_list() does not export well to ONNX
        #     # call _onnx_nested_tensor_from_tensor_list() instead
        #     return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

def get_config(conf):
    with open(conf, 'r') as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)

def print_config(conf):
    print(yaml.dump(conf, default_flow_style=False, default_style=''))

def read_txt(txt_path):
    f_read = open(txt_path, 'r')
    lines = f_read.readlines()

    out = list()
    for line in lines:
        out.append(line.rstrip())

    return out

def _get_box_color(class_name):
    # TODO: create color code
    box_color = (10, 180, 10)
    return box_color

def _load_weights(weights_dict):
    key, value = list(weights_dict.items())[0]

    trained_data_parallel = False
    if key[:7] == 'module.':
        trained_data_parallel = True

    if trained_data_parallel is True:
        new_weights = collections.OrderedDict()
        for old_key in weights_dict:
            new_key = old_key[7:]
            new_weights[new_key] = weights_dict[old_key]
    else:
        new_weights = weights_dict

    return new_weights

def _write_results(class_preds, bbox_preds, img_paths, class_idx_map, input_size, out_dir):
    class_preds_label = class_preds.argmax(dim=2)

    class_preds = class_preds.detach().cpu().tolist()
    class_preds_label = class_preds_label.detach().cpu().tolist()
    bbox_preds = bbox_preds.detach().cpu().tolist()

    for iter_batch, img_path in enumerate(img_paths):
        img_name = os.path.basename(img_path)
        out_path = os.path.join(out_dir, img_name.replace('.jpg', '.txt'))
        f_out = open(out_path, 'w')

        img = cv2.imread(img_path)
        resized_rows = input_size[0]
        resized_cols = input_size[1]
        ori_rows = img.shape[0]
        ori_cols = img.shape[1]
        ws = ori_cols / resized_cols
        hs = ori_rows / resized_rows

        class_pred = class_preds[iter_batch]
        class_pred_label = class_preds_label[iter_batch]
        bbox_pred = bbox_preds[iter_batch]

        for box_idx, box_label in enumerate(class_pred_label):
            ''' background idx is 0 '''
            if box_label == 0:
                continue

            box = bbox_pred[box_idx]

            ''' draw box '''
            pt1 = (int(box[0] * ws), int(box[1] * hs))
            pt2 = (int(box[2] * ws), int(box[3] * hs))
            class_name = class_idx_map[box_label]
            score = float(class_pred[box_idx][box_label])
            out_text = class_name + ':' + format(score, ".2f")
            box_color = _get_box_color(class_name)
            cv2.rectangle(img, pt1, pt2, box_color, 1)
            t_size = cv2.getTextSize(out_text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            font_pt2 = pt1[0] + (t_size[0] + 3), pt1[1] - (t_size[1] + 4)
            cv2.rectangle(img, pt1, font_pt2, box_color, -1)
            cv2.putText(img, out_text, (pt1[0], pt1[1] - (t_size[1] - 7)), cv2.FONT_HERSHEY_PLAIN, 1,
                        [225, 255, 255], 1)

            ''' write the result '''
            out_txt = str(class_name) + '\t' + \
                      str(pt1[0]) + '\t' + str(pt1[1]) + '\t' + str(pt2[0]) + '\t' + str(pt2[1]) + '\t' \
                      + str(score) + '\n'
            f_out.write(out_txt)

        cv2.imwrite(os.path.join(out_dir, img_name), img)
        f_out.close()



