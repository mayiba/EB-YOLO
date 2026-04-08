import os
import cv2
import numpy as np
from ultralytics import YOLO


# ===================== 配置区域 =====================
model_path = "/amax/tyut/user/zwk/zlw/glass/final_yolo/ultralytics-yolo11-main/runs/train/exp1_yolo11_bifpn_GLSA_LSCD_LQE4/weights/best.pt"
img_path = "/amax/tyut/user/zwk/zlw/glass/dataset_demo/images/train/1.jpg"
label_path = "/amax/tyut/user/zwk/zlw/glass/dataset_demo/labels/train/1.txt"
save_path = "./vis_tpfpfn_obb.jpg"

device = 3
conf_thres = 0.25
iou_thres = 0.5
# ===================================================


# ---------- 工具函数 ----------
def xyxyxyxy_to_rotated(box8):
    """
    8点 -> cv2 rotatedRect
    box8: (8,) [x1,y1,...,x4,y4]
    """
    pts = box8.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    return rect  # ((cx,cy),(w,h),angle)


def obb_iou(rect1, rect2):
    """
    rect: ((cx,cy),(w,h),angle)
    """
    inter_type, inter_pts = cv2.rotatedRectangleIntersection(rect1, rect2)
    if inter_type == 0 or inter_pts is None:
        return 0.0

    inter_area = cv2.contourArea(inter_pts)
    area1 = rect1[1][0] * rect1[1][1]
    area2 = rect2[1][0] * rect2[1][1]
    return inter_area / (area1 + area2 - inter_area + 1e-6)


def draw_rect(img, rect, color, thickness=2):
    pts = cv2.boxPoints(rect)
    pts = np.int32(pts)
    cv2.polylines(img, [pts], True, color, thickness)


# ---------- 主流程 ----------
def main():
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # ========== 读取 GT ==========
    gt_boxes = []
    gt_cls = []

    with open(label_path, "r") as f:
        for line in f.readlines():
            data = list(map(float, line.strip().split()))
            cls = int(data[0])
            pts = np.array(data[1:], dtype=np.float32).reshape(4, 2)

            # 如果你的 label 是像素坐标，注释掉下面两行
            pts[:, 0] *= w
            pts[:, 1] *= h

            rect = cv2.minAreaRect(pts)
            gt_boxes.append(rect)
            gt_cls.append(cls)

    gt_used = [False] * len(gt_boxes)

    # ========== 模型预测 ==========
    model = YOLO(model_path)
    results = model(img_path, device=device, conf=conf_thres, verbose=False)[0]

    pred_boxes = []
    pred_cls = []

    if results.obb is not None and len(results.obb) > 0:
        obb = results.obb.cpu()
        boxes = obb.xywhr.numpy()
        clss = obb.cls.numpy()

        for box, cls in zip(boxes, clss):
            cx, cy, bw, bh, angle = box
            rect = ((cx, cy), (bw, bh), angle * 180 / np.pi)
            pred_boxes.append(rect)
            pred_cls.append(int(cls))

    pred_used = [False] * len(pred_boxes)

    # ========== TP / FN ==========
    for i, (gt_rect, gt_c) in enumerate(zip(gt_boxes, gt_cls)):
        best_iou = 0
        best_j = -1

        for j, (pr_rect, pr_c) in enumerate(zip(pred_boxes, pred_cls)):
            if pred_used[j] or pr_c != gt_c:
                continue

            iou = obb_iou(gt_rect, pr_rect)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_thres:
            # TP
            pred_used[best_j] = True
            gt_used[i] = True
            draw_rect(img, pred_boxes[best_j], (0, 255, 0), 2)
        else:
            # FN
            draw_rect(img, gt_rect, (255, 0, 0), 2)

    # ========== FP ==========
    for j, used in enumerate(pred_used):
        if not used:
            draw_rect(img, pred_boxes[j], (0, 0, 255), 2)

    cv2.imwrite(save_path, img)
    print(f"✅ TP/FP/FN OBB 可视化完成: {save_path}")


if __name__ == "__main__":
    main()
