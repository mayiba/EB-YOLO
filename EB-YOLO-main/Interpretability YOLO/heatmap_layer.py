import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import os, shutil, copy
import cv2
import numpy as np
import torch
from tqdm import trange
from PIL import Image

from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression

from pytorch_grad_cam import (
    GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM,
    RandomCAM, EigenGradCAM, KPCA_CAM, AblationCAM
)
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114),
              auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (top, bottom, left, right)


class MyActivationsAndGradients:
    """Extract activations and register gradients from targeted intermediate layers."""

    def __init__(self, model, target_layers, reshape_transform=None):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []

        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))
            # do NOT use backward hook due to pytorch issue; use forward + tensor hook
            self.handles.append(target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, inp, out):
        activation = out
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.detach().cpu())

    def save_gradient(self, module, inp, out):
        if not hasattr(out, "requires_grad") or not out.requires_grad:
            return

        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.detach().cpu()] + self.gradients

        out.register_hook(_store_grad)

    def post_process(self, result):
        # NOTE: 这里保持你的处理逻辑不变
        if self.model.end2end:
            logits_ = result[:, :, 4:]
            boxes_ = result[:, :, :4]
            sorted_vals, indices = torch.sort(logits_[:, :, 0], descending=True)
            return logits_[0][indices[0]], boxes_[0][indices[0]]

        if self.model.task == 'detect':
            logits_ = result[:, 4:]
            boxes_ = result[:, :4]
            sorted_vals, indices = torch.sort(logits_.max(1)[0], descending=True)
            return (
                torch.transpose(logits_[0], 0, 1)[indices[0]],
                torch.transpose(boxes_[0], 0, 1)[indices[0]]
            )

        elif self.model.task == 'segment':
            logits_ = result[0][:, 4:4 + self.model.nc]
            boxes_ = result[0][:, :4]
            mask_p, mask_nm = result[1][2].squeeze(), result[1][1].squeeze().transpose(1, 0)
            c, h, w = mask_p.size()
            mask = (mask_nm @ mask_p.view(c, -1))
            sorted_vals, indices = torch.sort(logits_.max(1)[0], descending=True)
            return (
                torch.transpose(logits_[0], 0, 1)[indices[0]],
                torch.transpose(boxes_[0], 0, 1)[indices[0]],
                mask[indices[0]]
            )

        elif self.model.task == 'pose':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            poses_ = result[:, 4 + self.model.nc:]
            sorted_vals, indices = torch.sort(logits_.max(1)[0], descending=True)
            return (
                torch.transpose(logits_[0], 0, 1)[indices[0]],
                torch.transpose(boxes_[0], 0, 1)[indices[0]],
                torch.transpose(poses_[0], 0, 1)[indices[0]]
            )

        elif self.model.task == 'obb':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            angles_ = result[:, 4 + self.model.nc:]
            sorted_vals, indices = torch.sort(logits_.max(1)[0], descending=True)
            return (
                torch.transpose(logits_[0], 0, 1)[indices[0]],
                torch.transpose(boxes_[0], 0, 1)[indices[0]],
                torch.transpose(angles_[0], 0, 1)[indices[0]]
            )

        elif self.model.task == 'classify':
            return result[0]

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)

        if self.model.task == 'detect':
            post_result, pre_post_boxes = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes]]

        elif self.model.task == 'segment':
            post_result, pre_post_boxes, pre_post_mask = self.post_process(model_output)
            return [[post_result, pre_post_boxes, pre_post_mask]]

        elif self.model.task == 'pose':
            post_result, pre_post_boxes, pre_post_pose = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes, pre_post_pose]]

        elif self.model.task == 'obb':
            post_result, pre_post_boxes, pre_post_angle = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes, pre_post_angle]]

        elif self.model.task == 'classify':
            data = self.post_process(model_output)
            return [data]

    def release(self):
        for h in self.handles:
            h.remove()


class yolo_detect_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio, end2end) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
        self.end2end = end2end

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if (self.end2end and float(post_result[i, 0]) < self.conf) or \
               (not self.end2end and float(post_result[i].max()) < self.conf):
                break

            if self.ouput_type in ['class', 'all']:
                if self.end2end:
                    result.append(post_result[i, 0])
                else:
                    result.append(post_result[i].max())

            if self.ouput_type in ['box', 'all']:
                for j in range(4):
                    result.append(pre_post_boxes[i, j])

        return sum(result) if len(result) else post_result.sum() * 0.0


class yolo_segment_target(yolo_detect_target):
    def forward(self, data):
        post_result, pre_post_boxes, pre_post_mask = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type in ['class', 'all']:
                result.append(post_result[i].max())
            if self.ouput_type in ['box', 'all']:
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            if self.ouput_type in ['segment', 'all']:
                result.append(pre_post_mask[i].mean())
        return sum(result) if len(result) else post_result.sum() * 0.0


class yolo_pose_target(yolo_detect_target):
    def forward(self, data):
        post_result, pre_post_boxes, pre_post_pose = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type in ['class', 'all']:
                result.append(post_result[i].max())
            if self.ouput_type in ['box', 'all']:
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            if self.ouput_type in ['pose', 'all']:
                result.append(pre_post_pose[i].mean())
        return sum(result) if len(result) else post_result.sum() * 0.0


class yolo_obb_target(yolo_detect_target):
    def forward(self, data):
        post_result, pre_post_boxes, pre_post_angle = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type in ['class', 'all']:
                result.append(post_result[i].max())
            if self.ouput_type in ['box', 'all']:
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            if self.ouput_type in ['obb', 'all']:
                result.append(pre_post_angle[i])
        return sum(result) if len(result) else post_result.sum() * 0.0


class yolo_classify_target(yolo_detect_target):
    def forward(self, data):
        return data.max()


class yolo_heatmap:
    def __init__(self, weight, device, method, layer, backward_type,
                 conf_threshold, ratio, show_result, renormalize, task, img_size):

        self.device = torch.device(device)
        self.model_yolo = YOLO(weight)
        self.model_names = self.model_yolo.names
        print(f'model class info: {self.model_names}')

        self.model = copy.deepcopy(self.model_yolo.model).to(self.device)
        self.model.info()
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.eval()

        self.model.task = task
        if not hasattr(self.model, 'end2end'):
            self.model.end2end = False

        self.method_name = method
        self.cam_class = eval(method)  # GradCAMPlusPlus / GradCAM / ...
        self.layer = layer if isinstance(layer, (list, tuple)) else [layer]

        self.backward_type = backward_type
        self.conf_threshold = conf_threshold
        self.ratio = ratio
        self.show_result = show_result
        self.renormalize = renormalize
        self.task = task
        self.img_size = img_size

        if task == 'detect':
            self.target = yolo_detect_target(backward_type, conf_threshold, ratio, self.model.end2end)
        elif task == 'segment':
            self.target = yolo_segment_target(backward_type, conf_threshold, ratio, self.model.end2end)
        elif task == 'pose':
            self.target = yolo_pose_target(backward_type, conf_threshold, ratio, self.model.end2end)
        elif task == 'obb':
            self.target = yolo_obb_target(backward_type, conf_threshold, ratio, self.model.end2end)
        elif task == 'classify':
            self.target = yolo_classify_target(backward_type, conf_threshold, ratio, self.model.end2end)
        else:
            raise ValueError(f"not support task({task}).")

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize CAM inside each bbox; zero outside."""
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2 = min(grayscale_cam.shape[1] - 1, x2)
            y2 = min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(
                grayscale_cam[y1:y2, x1:x2].copy()
            )
        renormalized_cam = scale_cam_image(renormalized_cam)
        return show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)

    @torch.no_grad()
    def _predict_once(self, tensor):
        # ultralytics predict; tensor is (1,3,H,W) float
        pred = self.model_yolo.predict(tensor, conf=self.conf_threshold, iou=0.7)[0]
        return pred

    def process(self, img_path, save_dir):
        # read
        try:
            img_bgr = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            print(f"Warning... {img_path} read failure.")
            return

        img_lb, _, (top, bottom, left, right) = letterbox(
            img_bgr, new_shape=(self.img_size, self.img_size), auto=True
        )
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img_float = np.float32(img_rgb) / 255.0
        tensor = torch.from_numpy(np.transpose(img_float, (2, 0, 1))).unsqueeze(0).to(self.device)

        # 先跑一次预测（用于画框、renormalize）
        pred = self.model_yolo.predict(tensor, conf=self.conf_threshold, iou=0.7)[0]

        # 每层单独做 CAM
        for l in self.layer:
            if l < 0 or l >= len(self.model.model):
                print(f"Warning... layer index {l} out of range (0~{len(self.model.model)-1}). skip.")
                continue

            target_layers = [self.model.model[l]]
            cam_method = self.cam_class(self.model, target_layers)
            cam_method.activations_and_grads = MyActivationsAndGradients(self.model, target_layers, None)

            try:
                grayscale_cam = cam_method(tensor, [self.target])
            except AttributeError:
                print(f"Warning... CAM failure on layer {l}.")
                cam_method.activations_and_grads.release()
                continue

            grayscale_cam = grayscale_cam[0, :]  # (H,W)
            cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

            if self.renormalize and self.task in ['detect', 'segment', 'pose']:
                boxes = pred.boxes.xyxy.cpu().detach().numpy().astype(np.int32)
                cam_image = self.renormalize_cam_in_bounding_boxes(boxes, img_float, grayscale_cam)

            if self.show_result:
                cam_image = pred.plot(
                    img=cam_image,
                    conf=True,
                    font_size=None,
                    line_width=None,
                    labels=False,
                )

            # remove padding
            cam_image = cam_image[top:cam_image.shape[0] - bottom,
                                  left:cam_image.shape[1] - right]

            out_path = os.path.join(save_dir, f'layer{l}.png')
            Image.fromarray(cam_image).save(out_path)

            # 释放hook，避免叠加
            cam_method.activations_and_grads.release()

    def __call__(self, img_path, save_path):
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            for img_name in os.listdir(img_path):
                in_path = os.path.join(img_path, img_name)
                # 每张图建一个子目录，避免不同图片的 layer 文件互相覆盖
                stem = os.path.splitext(img_name)[0]
                out_dir = os.path.join(save_path, stem)
                os.makedirs(out_dir, exist_ok=True)
                self.process(in_path, out_dir)
        else:
            self.process(img_path, save_path)


def get_params():
    return {
        'weight': '/amax/tyut/user/zwk/zlw/glass/final_yolo/ultralytics-yolo11-main/runs/train/exp1_yolo11_bifpn_GLSA_LSCD_LQE4/weights/best.pt',
        'device': 'cuda:3',
        'method': 'GradCAMPlusPlus',  # GradCAMPlusPlus, GradCAM, ...
        'layer': [10, 12, 14, 16, 18],
        'backward_type': 'all',
        'conf_threshold': 0.2,
        'ratio': 0.02,
        'show_result': True,
        'renormalize': False,
        'task': 'detect',
        'img_size': 640,
    }


if __name__ == '__main__':
    model = yolo_heatmap(**get_params())
    model(r'/amax/tyut/user/zwk/zlw/glass/dataset_demo/images/train/1.jpg', 'result_layer')
    # model(r'/path/to/dir', 'result')
