import torch
from safetensors.torch import load_file

# 1. 你的 safetensors 权重路径
safe_path = "/amax/tyut/user/zwk/zlw/glass/3yolo/ultralytics-yolo11-main/ultralytics/model3.safetensors"
# 2. 想导出的 pt 文件路径
pt_path = "/amax/tyut/user/zwk/zlw/glass/3yolo/ultralytics-yolo11-main/ultralytics/model3.pt"

# 读取 safetensors（返回的是一个 state_dict(dict[str, Tensor])）
sd = load_file(safe_path)

# 如果你的 convnextv2_xxx 代码里是用 torch.load(... )['model'] 这种形式，
# 那这里就把它包装成 {'model': sd}
torch.save({"model": sd}, pt_path)

print("已保存到:", pt_path)
