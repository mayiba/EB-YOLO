# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn.functional as F
# from ultralytics.nn.tasks import attempt_load_weights
#
# # ----------------------------
# # Utilities (same as your code)
# # ----------------------------
# def find_all_pcaa_modules(model):
#     pcaa_modules = []
#     for name, module in model.named_modules():
#         if module.__class__.__name__ == 'GLSA':
#             pcaa_modules.append((name, module))
#     return pcaa_modules
#
# def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
#     h, w = im.shape[:2]
#     r = min(new_shape[0] / h, new_shape[1] / w)
#     new_unpad = (int(w * r), int(h * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
#     dw /= 2
#     dh /= 2
#     im = cv2.resize(im, new_unpad)
#     im = cv2.copyMakeBorder(
#         im, int(dh), int(dh), int(dw), int(dw),
#         cv2.BORDER_CONSTANT, value=color
#     )
#     return im
#
# def preprocess(img_path, device):
#     img_bgr = cv2.imread(img_path)
#     img_bgr = letterbox(img_bgr)
#     img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     img = img.astype(np.float32) / 255.0
#     x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
#     return x, img
#
# # ----------------------------
# # PCAA attention recomputation
# # (no need to modify PCAA_CVPR2022.py)
# # ----------------------------
# def patch_split(input, bin_size):
#     """
#     b c (bh rh) (bw rw) -> b (bh bw) rh rw c
#     """
#     B, C, H, W = input.size()
#     bin_num_h, bin_num_w = bin_size
#     rH = H // bin_num_h
#     rW = W // bin_num_w
#     out = input.view(B, C, bin_num_h, rH, bin_num_w, rW)
#     out = out.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, bh, bw, rH, rW, C]
#     out = out.view(B, -1, rH, rW, C)                   # [B, bins, rH, rW, C]
#     return out
#
# def _pad_to_bin(t: torch.Tensor, bin_size, pad_value: float = 0.0):
#     _, _, H, W = t.shape
#     bh, bw = bin_size
#     pad_h = (bh - H % bh) % bh
#     pad_w = (bw - W % bw) % bw
#     if pad_h != 0 or pad_w != 0:
#         t = F.pad(t, (0, pad_w, 0, pad_h), mode="constant", value=pad_value)
#     return t, pad_h, pad_w
#
# class PCAA_AttentionVisualizer:
#     """
#     Make PCAA visualization look like interpretability_analysis.py:
#     4 columns:
#       1) Spatial overlay (heatmap on image)
#       2) Spatial 3D surface
#       3) "Channel" distribution (scatter)  -> here we use per-class weights from bin_confidence
#       4) "Channel" weight (bar)            -> same weights bar
#     """
#     def __init__(self, pcaa_module):
#         self.pcaa = pcaa_module
#         self._latest_input = None
#         self._hook = None
#
#     def _install_pre_hook(self):
#         def pre_hook(module, inputs):
#             x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
#             self._latest_input = x.detach()
#         self._hook = self.pcaa.register_forward_pre_hook(pre_hook)
#
#     def _remove_hook(self):
#         if self._hook is not None:
#             self._hook.remove()
#             self._hook = None
#
#     @torch.no_grad()
#     def run_and_capture_input(self, model, x):
#         self._latest_input = None
#         self._install_pre_hook()
#         _ = model(x)  # trigger pre_hook
#         self._remove_hook()
#         if self._latest_input is None:
#             raise RuntimeError("Failed to capture PCAA input via forward_pre_hook.")
#         return self._latest_input
#
#     @torch.no_grad()
#     def recompute_pcaa_attn(self, x_in):
#         """
#         Recompute key attention tensors from PCAA, matching your PCAA_CVPR2022.py logic:
#           - pixel_confidence: softmax(cam_s, dim=2)  [B, bins, rH*rW, K]
#           - bin_confidence:  cls_score reshaped      [B, bins, K, 1]
#           - rH, rW
#         """
#         pcaa = self.pcaa
#         B, C, H0, W0 = x_in.shape
#
#         cam = pcaa.conv_cam(pcaa.dropout(x_in))         # [B, K, H, W]
#         cls_score = pcaa.sigmoid(pcaa.pool_cam(cam))    # [B, K, bh, bw]
#
#         x_pad, _, _ = _pad_to_bin(x_in, pcaa.bin_size, pad_value=0.0)
#         cam_pad, _, _ = _pad_to_bin(cam, pcaa.bin_size, pad_value=-1e4)
#
#         cam_s = patch_split(cam_pad, pcaa.bin_size)     # [B, bins, rH, rW, K]
#         x_s = patch_split(x_pad, pcaa.bin_size)         # [B, bins, rH, rW, C]
#
#         rH, rW = cam_s.shape[2], cam_s.shape[3]
#         K = cam_s.shape[-1]
#
#         cam_s = cam_s.view(B, -1, rH * rW, K)           # [B, bins, rH*rW, K]
#         # bin_confidence: [B, bins, K, 1]
#         bin_confidence = cls_score.view(B, K, -1).transpose(1, 2).unsqueeze(3)
#
#         # pixel_confidence over pixels within each bin
#         pixel_confidence = F.softmax(cam_s, dim=2)
#
#         return {
#             "pixel_confidence": pixel_confidence,  # [B, bins, rH*rW, K]
#             "bin_confidence": bin_confidence,      # [B, bins, K, 1]
#             "rH": rH,
#             "rW": rW
#         }
#
#     def _normalize(self, a: np.ndarray):
#         a = a - a.min()
#         return a / (a.max() + 1e-8)
#
#     def visualize_like_interpretability(self, img_rgb, attn_dict, save_path):
#         """
#         img_rgb: HxWx3 float [0..1]
#         """
#         pixel_conf = attn_dict["pixel_confidence"]  # [B, bins, rH*rW, K]
#         bin_conf = attn_dict["bin_confidence"]      # [B, bins, K, 1]
#         rH, rW = attn_dict["rH"], attn_dict["rW"]
#
#         # ---- Spatial map (like spatial attention) ----
#         # average over bins and classes(K): -> [B, rH*rW]
#         spatial_1d = pixel_conf.mean(dim=(1, 3))[0]      # [rH*rW]
#         spatial_2d = spatial_1d.view(rH, rW).cpu().numpy()
#         spatial_2d = self._normalize(spatial_2d)
#
#         H, W = img_rgb.shape[:2]
#         spatial_resized = cv2.resize(spatial_2d, (W, H), interpolation=cv2.INTER_CUBIC)
#
#         # ---- "Channel" weights (use per-class weight from bin_confidence) ----
#         # bin_conf: [B, bins, K, 1] -> avg over bins -> [K]
#         channel_w = bin_conf.mean(dim=1).squeeze(-1)[0].cpu().numpy()  # [K]
#         channel_w = self._normalize(channel_w)
#
#         # ---- Build 1x4 layout ----
#         from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
#
#         fig = plt.figure(figsize=(20, 5))
#
#         # col1: overlay heatmap
#         ax1 = fig.add_subplot(1, 4, 1)
#         ax1.imshow(img_rgb)
#         im1 = ax1.imshow(spatial_resized, cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)
#         ax1.set_title("PCAA\nSpatial Attention (Heatmap)")
#         ax1.axis("off")
#         cbar1 = plt.colorbar(im1, ax=ax1)
#         cbar1.set_label("Attention Value", rotation=270, labelpad=15)
#
#         # col2: 3D surface
#         ax2 = fig.add_subplot(1, 4, 2, projection="3d")
#         surf_map = spatial_resized
#         # downsample for speed
#         max_size = 120
#         if max(surf_map.shape) > max_size:
#             scale = max_size / max(surf_map.shape)
#             new_w = int(surf_map.shape[1] * scale)
#             new_h = int(surf_map.shape[0] * scale)
#             surf_map = cv2.resize(surf_map, (new_w, new_h), interpolation=cv2.INTER_AREA)
#
#         hh, ww = surf_map.shape
#         X, Y = np.meshgrid(np.arange(ww), np.arange(hh))
#         Z = np.fliplr(surf_map)  # match interpretability_analysis.py trick
#         surf = ax2.plot_surface(X, Y, Z, cmap="jet", vmin=0.0, vmax=1.0,
#                                 linewidth=0, antialiased=True, alpha=0.9, shade=True)
#         ax2.set_title("PCAA\nSpatial Attention (Weight)")
#         ax2.set_xlabel("Width")
#         ax2.set_ylabel("Height")
#         ax2.set_zlabel("Attention Weight")
#         ax2.view_init(elev=30, azim=45)
#         ax2.invert_xaxis()
#         fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=20, label="Attention Weight")
#
#         # col3: scatter distribution
#         ax3 = fig.add_subplot(1, 4, 3)
#         idxs = np.arange(len(channel_w))
#         scatter = ax3.scatter(
#             idxs, channel_w,
#             c=channel_w, s=50 + channel_w * 200,
#             cmap="hot_r", alpha=0.7,
#             edgecolors="black", linewidths=0.5,
#             vmin=0.0, vmax=1.0
#         )
#         ax3.set_title("PCAA\nChannel Attention (Distribution)")
#         ax3.set_xlabel("Class/Channel Index (K)")
#         ax3.set_ylabel("Attention Weight")
#         ax3.set_ylim(0.0, 1.0)
#         ax3.grid(True, alpha=0.3)
#         cbar3 = plt.colorbar(scatter, ax=ax3)
#         cbar3.set_label("Attention Weight", rotation=270, labelpad=15)
#
#         # col4: bar weights
#         ax4 = fig.add_subplot(1, 4, 4)
#         ax4.bar(idxs, channel_w, alpha=0.8)
#         ax4.set_title("PCAA\nChannel Attention (Weight)")
#         ax4.set_xlabel("Class/Channel Index (K)")
#         ax4.set_ylabel("Attention Weight")
#         ax4.set_ylim(0.0, 1.0)
#
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=150, bbox_inches="tight")
#         plt.close()
#
#
# def main():
#     # weight = "/amax/tyut/user/zwk/zlw/glass/3yolo/ultralytics-yolo11-main/runs/train/exp1_yolo8_PCAA5/weights/best.pt"
#     weight = "/amax/tyut/user/zwk/zlw/glass/final_yolo/ultralytics-yolo11-main/runs/train/exp1_yolo11_bifpn_GLSA_LSCD_LQE3/weights/best.pt"
#     img_path = "/amax/tyut/user/zwk/zlw/glass/dataset_demo/images/train/1.jpg"
#     device = torch.device("cuda:3")
#
#     save_root = "result"
#     os.makedirs(save_root, exist_ok=True)
#
#     model = attempt_load_weights(weight, device)
#     model.eval()
#
#     print("===== Model Modules (Top-Level) =====")
#     for i, m in enumerate(model.model):
#         print(i, m.__class__.__name__)
#
#     x, img = preprocess(img_path, device)
#
#     pcaa_list = find_all_pcaa_modules(model)
#     print(f"Found {len(pcaa_list)} GLSA modules:")
#     for name, _ in pcaa_list:
#         print("  ", name)
#
#     for idx, (name, pcaa) in enumerate(pcaa_list):
#         vis = PCAA_AttentionVisualizer(pcaa)
#
#         # 1) run full model once and capture the input feature of this PCAA
#         x_in = vis.run_and_capture_input(model, x)
#
#         # 2) recompute attention tensors from that input feature
#         attn = vis.recompute_pcaa_attn(x_in)
#
#         # 3) visualize with unified 4-col style
#         save_path = f"{save_root}/GLSA_{idx}_{name.replace('.', '_')}.png"
#         vis.visualize_like_interpretability(img, attn, save_path)
#         print(f"[Saved] {save_path}")
#
#
# if __name__ == "__main__":
#     main()
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from ultralytics.nn.tasks import attempt_load_weights


# ----------------------------
# Image preprocess (robust letterbox)
# ----------------------------
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """YOLO-style letterbox with correct padding split (top/bottom, left/right)."""
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    left = int(np.floor(dw / 2))
    right = int(np.ceil(dw / 2))
    top = int(np.floor(dh / 2))
    bottom = int(np.ceil(dh / 2))

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im

def preprocess(img_path, device, size=(640, 640)):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_bgr = letterbox(img_bgr, new_shape=size)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return x, img


# ----------------------------
# Module discovery (attention-name-agnostic)
# ----------------------------
def find_modules_by_classname(model, classnames=("GLSA",)):
    found = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in classnames:
            found.append((name, module))
    return found


# ----------------------------
# Attention extraction (hook + recompute/approx)
# ----------------------------
class AttentionExtractor:
    """
    Like interpretability_analysis.py:
    - forward hook caches input/output feature for each target module
    - then recompute "spatial" & "channel" weights
      * if module exposes explicit attention map: use it
      * else fallback to activation-based attention (mean abs)
    """
    def __init__(self, model, target_modules):
        self.model = model
        self.target_modules = target_modules  # list[(name, module)]

        self.inputs = {}   # name -> tensor [B,C,H,W]
        self.outputs = {}  # name -> tensor [B,C,H,W] or other
        self.extra = {}    # name -> dict (maybe explicit attn)
        self.hooks = []

    def _hook_fn(self, name, module, inputs, output):
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        self.inputs[name] = x.detach().float().cpu()

        # output can be tensor or tuple/list
        self.outputs[name] = (
            output[0].detach().float().cpu()
            if isinstance(output, (tuple, list)) and torch.is_tensor(output[0])
            else (output.detach().float().cpu() if torch.is_tensor(output) else None)
        )

        # try to find explicit attention if module keeps it
        info = {}
        for key in ["attn", "attention", "attn_map", "attention_map", "weights"]:
            if hasattr(module, key):
                v = getattr(module, key)
                if torch.is_tensor(v):
                    info[key] = v.detach().float().cpu()
        # or if output is (y, attn)
        if isinstance(output, (tuple, list)) and len(output) >= 2 and torch.is_tensor(output[1]):
            info["output_attn"] = output[1].detach().float().cpu()

        if info:
            self.extra[name] = info

    def register(self):
        for name, module in self.target_modules:
            h = module.register_forward_hook(lambda m, i, o, n=name: self._hook_fn(n, m, i, o))
            self.hooks.append(h)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    @torch.no_grad()
    def run(self, x):
        self.inputs.clear()
        self.outputs.clear()
        self.extra.clear()
        self.register()
        _ = self.model(x)
        self.remove()

    def compute_attention(self, prefer_output=True):
        """
        Returns:
          maps[name] = {"spatial": [H,W], "channel": [C]}
        """
        maps = {}
        for name, _ in self.target_modules:
            feat = None
            if prefer_output and name in self.outputs and self.outputs[name] is not None:
                feat = self.outputs[name]
            elif name in self.inputs:
                feat = self.inputs[name]

            if feat is None or feat.ndim != 4:
                continue

            # feat: [B,C,H,W]
            f = feat[0]  # first batch

            # --- explicit attention? (optional) ---
            # if you later confirm GLSA stores something like module.attn, you can use it here.
            # For now: we keep it simple and robust.

            # --- activation-based spatial ---
            spatial = torch.mean(torch.abs(f), dim=0)          # [H,W]
            spatial = spatial - spatial.min()
            spatial = spatial / (spatial.max() + 1e-8)

            # --- activation-based channel ---
            channel = torch.mean(torch.abs(f), dim=(1, 2))     # [C]
            channel = channel - channel.min()
            channel = channel / (channel.max() + 1e-8)

            maps[name] = {
                "spatial": spatial.numpy(),
                "channel": channel.numpy(),
            }
        return maps


# ----------------------------
# Visualization (4 columns, global scaling)
# ----------------------------
class AttentionVisualizer:
    def __init__(self, save_root="result"):
        self.save_root = save_root
        os.makedirs(save_root, exist_ok=True)

    @staticmethod
    def _resize_map(m, W, H):
        return cv2.resize(m, (W, H), interpolation=cv2.INTER_CUBIC)

    def _collect_global_ranges(self, img_rgb, maps):
        H, W = img_rgb.shape[:2]
        spatial_vals = []
        channel_vals = []
        for d in maps.values():
            if "spatial" in d and d["spatial"] is not None:
                spatial_vals.append(self._resize_map(d["spatial"], W, H).reshape(-1))
            if "channel" in d and d["channel"] is not None:
                channel_vals.append(d["channel"].reshape(-1))

        if spatial_vals:
            spatial_all = np.concatenate(spatial_vals)
            smin, smax = float(spatial_all.min()), float(spatial_all.max())
        else:
            smin, smax = 0.0, 1.0

        if channel_vals:
            channel_all = np.concatenate(channel_vals)
            cmin, cmax = float(channel_all.min()), float(channel_all.max())
        else:
            cmin, cmax = 0.0, 1.0
        return (smin, smax), (cmin, cmax)

    def visualize(self, img_rgb, maps, tag="attention"):
        """
        One row per module, 4 columns:
          1) spatial overlay
          2) spatial 3D surface
          3) channel scatter
          4) channel bar
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        H, W = img_rgb.shape[:2]
        (smin, smax), (cmin, cmax) = self._collect_global_ranges(img_rgb, maps)

        names = list(maps.keys())
        n = len(names)
        if n == 0:
            print("[Warn] No valid feature maps to visualize.")
            return

        fig = plt.figure(figsize=(20, 5 * n))

        for row, name in enumerate(names, start=1):
            spatial = maps[name].get("spatial", None)
            channel = maps[name].get("channel", None)

            # col1: overlay
            ax1 = fig.add_subplot(n, 4, (row - 1) * 4 + 1)
            ax1.imshow(img_rgb)
            if spatial is not None:
                sp = self._resize_map(spatial, W, H)
                im1 = ax1.imshow(sp, cmap="jet", alpha=0.5, vmin=smin, vmax=smax)
                ax1.set_title(f"{name}\nSpatial (Overlay)")
                ax1.axis("off")
                cbar1 = plt.colorbar(im1, ax=ax1)
                cbar1.set_label("Weight", rotation=270, labelpad=15)
            else:
                ax1.set_title(f"{name}\nSpatial (None)")
                ax1.axis("off")

            # col2: 3D surface
            ax2 = fig.add_subplot(n, 4, (row - 1) * 4 + 2, projection="3d")
            if spatial is not None:
                sp = self._resize_map(spatial, W, H)

                # downsample for speed
                max_size = 120
                if max(sp.shape) > max_size:
                    scale = max_size / max(sp.shape)
                    new_w = int(sp.shape[1] * scale)
                    new_h = int(sp.shape[0] * scale)
                    sp = cv2.resize(sp, (new_w, new_h), interpolation=cv2.INTER_AREA)

                hh, ww = sp.shape
                X, Y = np.meshgrid(np.arange(ww), np.arange(hh))
                Z = np.fliplr(sp)
                surf = ax2.plot_surface(X, Y, Z, cmap="jet", vmin=smin, vmax=smax,
                                        linewidth=0, antialiased=True, alpha=0.9, shade=True)
                ax2.set_title(f"{name}\nSpatial (3D)")
                ax2.set_xlabel("Width")
                ax2.set_ylabel("Height")
                ax2.set_zlabel("Weight")
                ax2.view_init(elev=30, azim=45)
                ax2.invert_xaxis()
                fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=20, label="Weight")
            else:
                ax2.set_title(f"{name}\nSpatial (None)")

            # col3: channel scatter
            ax3 = fig.add_subplot(n, 4, (row - 1) * 4 + 3)
            if channel is not None:
                idxs = np.arange(len(channel))
                scatter = ax3.scatter(
                    idxs, channel,
                    c=channel, s=50 + channel * 200,
                    cmap="hot_r", alpha=0.7,
                    edgecolors="black", linewidths=0.5,
                    vmin=cmin, vmax=cmax
                )
                ax3.set_title(f"{name}\nChannel (Scatter)")
                ax3.set_xlabel("Index")
                ax3.set_ylabel("Weight")
                ax3.set_ylim(cmin, cmax if cmax > cmin else 1.0)
                ax3.grid(True, alpha=0.3)
                cbar3 = plt.colorbar(scatter, ax=ax3)
                cbar3.set_label("Weight", rotation=270, labelpad=15)
            else:
                ax3.set_title(f"{name}\nChannel (None)")
                ax3.axis("off")

            # col4: channel bar
            ax4 = fig.add_subplot(n, 4, (row - 1) * 4 + 4)
            if channel is not None:
                idxs = np.arange(len(channel))
                ax4.bar(idxs, channel, alpha=0.8)
                ax4.set_title(f"{name}\nChannel (Bar)")
                ax4.set_xlabel("Index")
                ax4.set_ylabel("Weight")
                ax4.set_ylim(cmin, cmax if cmax > cmin else 1.0)
            else:
                ax4.set_title(f"{name}\nChannel (None)")
                ax4.axis("off")

        plt.tight_layout()
        save_path = os.path.join(self.save_root, f"{tag}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {save_path}")


def main():
    weight = "/amax/tyut/user/zwk/zlw/glass/final_yolo/ultralytics-yolo11-main/runs/train/exp1_yolo11_bifpn_GLSA_LSCD_LQE4/weights/best.pt"
    img_path = "/amax/tyut/user/zwk/zlw/glass/dataset_demo/images/train/1.jpg"
    device = torch.device("cuda:3")

    save_root = "result"
    os.makedirs(save_root, exist_ok=True)

    model = attempt_load_weights(weight, device)
    model.eval()

    x, img = preprocess(img_path, device)

    targets = find_modules_by_classname(model, classnames=("GLSA",))
    print(f"Found {len(targets)} target modules:")
    for name, m in targets:
        print(" ", name, "->", m.__class__.__name__)

    extractor = AttentionExtractor(model, targets)
    extractor.run(x)
    maps = extractor.compute_attention(prefer_output=True)

    viz = AttentionVisualizer(save_root=save_root)
    viz.visualize(img, maps, tag="module_attention_summary")


if __name__ == "__main__":
    main()
