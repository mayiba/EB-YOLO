import torch.nn as nn
from torch.nn import functional as F
import torch

from ultralytics.nn.modules import C3


# https://github.com/lsa1997/PCAA/blob/main/networks/caanet.py
#https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Partial_Class_Activation_Attention_for_Semantic_Segmentation_CVPR_2022_paper.pdf

__all__ =['C2f_PCAA','C3k2_PCAA','PCAA']
def patch_split(input, bin_size):
    """
    b c (bh rh) (bw rw) -> b (bh bw) rh rw c
    """
    B, C, H, W = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    rH = H // bin_num_h
    rW = W // bin_num_w
    out = input.view(B, C, bin_num_h, rH, bin_num_w, rW)
    out = out.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, bin_num_h, bin_num_w, rH, rW, C]
    out = out.view(B, -1, rH, rW, C)  # [B, bin_num_h * bin_num_w, rH, rW, C]
    return out


def patch_recover(input, bin_size):
    """
    b (bh bw) rh rw c -> b c (bh rh) (bw rw)
    """
    B, N, rH, rW, C = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    H = rH * bin_num_h
    W = rW * bin_num_w
    out = input.view(B, bin_num_h, bin_num_w, rH, rW, C)
    out = out.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, bin_num_h, rH, bin_num_w, rW]
    out = out.view(B, C, H, W)  # [B, C, H, W]
    return out


class GCN(nn.Module):
    def __init__(self, num_node, num_channel):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv2d(num_node, num_node, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Linear(num_channel, num_channel, bias=False)

    def forward(self, x):
        # x: [B, bin_num_h * bin_num_w, K, C]
        out = self.conv1(x)
        out = self.relu(out + x)
        out = self.conv2(out)
        return out


class PCAA(nn.Module):
    def __init__(self, feat_in, bin_size=(4,4), norm_layer=nn.BatchNorm2d):
        super(PCAA, self).__init__()
        feat_inner = feat_in // 2
        num_classes = feat_in
        self.norm_layer = norm_layer
        self.bin_size = bin_size
        self.dropout = nn.Dropout2d(0.1)
        self.conv_cam = nn.Conv2d(feat_in, num_classes, kernel_size=1)
        self.pool_cam = nn.AdaptiveAvgPool2d(bin_size)
        self.sigmoid = nn.Sigmoid()

        bin_num = bin_size[0] * bin_size[1]
        self.gcn = GCN(bin_num, feat_in)
        self.fuse = nn.Conv2d(bin_num, 1, kernel_size=1)
        self.proj_query = nn.Linear(feat_in, feat_inner)
        self.proj_key = nn.Linear(feat_in, feat_inner)
        self.proj_value = nn.Linear(feat_in, feat_inner)

        self.conv_out = nn.Sequential(
            nn.Conv2d(feat_inner, feat_in, kernel_size=1, bias=False),
            norm_layer(feat_in),
            nn.ReLU(inplace=True)
        )
        self.scale = feat_inner ** -0.5
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _pad_to_bin(t: torch.Tensor, bin_size, pad_value: float = 0.0):
        """Pad right/bottom so H,W are divisible by bin_size."""
        _, _, H, W = t.shape
        bh, bw = bin_size
        pad_h = (bh - H % bh) % bh
        pad_w = (bw - W % bw) % bw
        # F.pad order: (left, right, top, bottom)
        if pad_h != 0 or pad_w != 0:
            t = F.pad(t, (0, pad_w, 0, pad_h), mode="constant", value=pad_value)
        return t, pad_h, pad_w

    def forward(self, x):
        # 原始尺寸（用于最终裁剪 + residual）
        residual = x
        B, C, H0, W0 = x.shape

        # cam + bin confidence 用原始 x 计算（不依赖 H,W 是否整除）
        cam = self.conv_cam(self.dropout(x))                  # [B, K, H, W]
        cls_score = self.sigmoid(self.pool_cam(cam))          # [B, K, bh, bw]

        # ======= 关键：pad 到可被 bin_size 整除 =======
        # x padding 用 0 就行；cam padding 用大负数，保证 softmax 后 padding 像素权重≈0
        x_pad, pad_h, pad_w = self._pad_to_bin(x, self.bin_size, pad_value=0.0)
        cam_pad, _, _ = self._pad_to_bin(cam, self.bin_size, pad_value=-1e4)

        # patch split
        cam_s = patch_split(cam_pad, self.bin_size)  # [B, bins, rH, rW, K]
        x_s = patch_split(x_pad, self.bin_size)      # [B, bins, rH, rW, C]

        rH = cam_s.shape[2]
        rW = cam_s.shape[3]
        K = cam_s.shape[-1]
        C = x_s.shape[-1]

        cam_s = cam_s.view(B, -1, rH * rW, K)  # [B, bins, rH*rW, K]
        x_s = x_s.view(B, -1, rH * rW, C)      # [B, bins, rH*rW, C]

        bin_confidence = cls_score.view(B, K, -1).transpose(1, 2).unsqueeze(3)  # [B, bins, K, 1]
        pixel_confidence = F.softmax(cam_s, dim=2)  # softmax over pixels in each bin

        local_feats = torch.matmul(pixel_confidence.transpose(2, 3), x_s) * bin_confidence  # [B, bins, K, C]
        local_feats = self.gcn(local_feats)                                                # [B, bins, K, C]
        global_feats = self.fuse(local_feats)                                              # [B, 1, K, C]
        global_feats = self.relu(global_feats).repeat(1, x_s.shape[1], 1, 1)                # [B, bins, K, C]

        query = self.proj_query(x_s)           # [B, bins, rH*rW, C//2]
        key   = self.proj_key(local_feats)     # [B, bins, K,     C//2]
        value = self.proj_value(global_feats)  # [B, bins, K,     C//2]

        aff_map = torch.matmul(query, key.transpose(2, 3))  # [B, bins, rH*rW, K]
        aff_map = F.softmax(aff_map, dim=-1)
        out = torch.matmul(aff_map, value)                  # [B, bins, rH*rW, C//2]

        out = out.view(B, -1, rH, rW, value.shape[-1])      # [B, bins, rH, rW, C//2]
        out = patch_recover(out, self.bin_size)             # [B, C//2, H_pad, W_pad]

        out = self.conv_out(out)                            # [B, C, H_pad, W_pad]

        # ======= 关键：裁剪回原始 H0,W0，再做残差 =======
        out = out[:, :, :H0, :W0].contiguous()
        out = residual + out
        return out

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Bottleneck_PCAA(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.Attention = PCAA(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))



class C2f_PCAA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_PCAA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_PCAA(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

class C3k2_PCAA(C2f_PCAA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_PCAA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.rand(1, 64, 128, 128)
    pcaa = PCAA(64)
    output = pcaa(input)
    print("PCAA_input.shape:", input.shape)
    print("PCAA_output.shape:",output.shape)




