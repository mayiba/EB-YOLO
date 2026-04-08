import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch
import cv2
import os
import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Serif'
import seaborn as sns

from ultralytics.nn.tasks import attempt_load_weights
from timm.utils import AverageMeter


# ========================= 工具函数 =========================
def get_activation(feat):
    def hook(model, inputs, outputs):
        feat.append(outputs)
    return hook


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114),
              auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im


def get_rectangle(data, thresh):
    h, w = data.shape
    total = data.sum()
    for i in range(1, h // 2):
        area = data[h//2-i:h//2+i+1, w//2-i:w//2+i+1]
        if area.sum() / total > thresh:
            side = i * 2 + 1
            return side, (side / h) * (side / w)
    return None, None


from matplotlib.colors import LinearSegmentedColormap

YG_CMAP = LinearSegmentedColormap.from_list(
    "YellowGreen",
    [
        (1.0, 1.0, 0.6),   # low contribution (yellow)
        (0.2, 0.8, 0.2)    # high contribution (green)
    ]
)

def draw_heatmap(data, save_path):
    plt.figure(figsize=(10, 10), dpi=300)
    sns.heatmap(
        data,
        cmap=YG_CMAP,
        xticklabels=False,
        yticklabels=False,
        cbar=True
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ========================= ERF 类 =========================
class YOLO_ERF:
    def __init__(self, weight, device, layer_idx, dataset, num_images, save_path):
        self.device = torch.device(device)
        self.dataset = dataset
        self.num_images = num_images
        self.save_path = save_path
        self.feature = []

        self.model = attempt_load_weights(weight, self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        self.meter = AverageMeter()

        # 注册 hook
        self.hook = self.model.model[int(layer_idx)].register_forward_hook(
            get_activation(self.feature)
        )

    def get_input_grad(self, samples):
        _ = self.model(samples)
        feat = self.feature[-1]
        self.feature.clear()

        h, w = feat.shape[2:]
        center = torch.relu(feat[:, :, h//2, w//2]).sum()
        grad = torch.autograd.grad(center, samples)[0]
        grad = torch.relu(grad)
        return grad.sum((0, 1)).cpu().numpy()

    def process(self):
        img_list = os.listdir(self.dataset)[:self.num_images]

        for idx, name in enumerate(img_list):
            img = cv2.imread(os.path.join(self.dataset, name))
            if img is None:
                continue

            img = letterbox(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0

            samples = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            samples.requires_grad = True

            self.optimizer.zero_grad()
            grad_map = self.get_input_grad(samples)

            if not np.isnan(grad_map).any():
                self.meter.update(grad_map)

            print(f"[Layer {self.save_path}] {idx+1}/{self.num_images}")

        data = self.meter.avg
        data = np.log10(data + 1)
        data = data / data.max()

        print(f"==== ERF statistics for {self.save_path} ====")
        for t in [0.2, 0.3, 0.5, 0.99]:
            side, ratio = get_rectangle(data, t)
            print(f"thresh={t}, side={side}, area_ratio={ratio}")

        draw_heatmap(data, self.save_path)
        self.hook.remove()


# ========================= 主入口 =========================
def main():
    weight = "/home/chenhm/PycharmProjects/PythonProject1/final_yolo/final_yolo/ultralytics-yolo11-main/runs/train/exp1_yolo11_bifpn_GLSA_LSCD_LQE6/weights/best.pt"
    dataset = "/home/chenhm/PycharmProjects/PythonProject1/ultralytics/dataset_demo/images/train"
    device = "cuda:0"
    num_images = 50

    layers = ['4', '6', '8', '10']  # 你可以自由加
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)

    for layer in layers:
        save_path = os.path.join(result_dir, f"erf_layer{layer}.png")
        erf = YOLO_ERF(
            weight=weight,
            device=device,
            layer_idx=layer,
            dataset=dataset,
            num_images=num_images,
            save_path=save_path
        )
        erf.process()


if __name__ == "__main__":
    main()
