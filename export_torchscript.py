import torch
from pathlib import Path
from torchvision import transforms, models
import torch.nn as nn

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else torch.device("cpu")
)

config = {
    'drop_out': 0.3
}
def init_model(num_of_classes=10):
    model = models.resnet18(pretrained=True)  # 加载预训练的ResNet18
    num_features = model.fc.in_features  # 获取原分类器的输入特征数
    # 替换分类器最后一层
    model.fc = nn.Sequential(
        nn.Dropout(config['drop_out']),  # 30%的dropout
        nn.Linear(num_features, num_of_classes)  # 全连接层
    )

    return model


# ===== 路径与目录 =====
ckpt_path = "checkpoints/pneumonia_model.pth"
outdir = Path("torchscript")
outdir.mkdir(parents=True, exist_ok=True)  # 关键：先建目录！
ts_path = outdir / "pneumonia_ts_model.pt"

# ===== 加载权重（尽量安全）=====
ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

# ckpt 可能是字典({..., "model_state_dict": ...})，也可能就是 state_dict
state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
# 若曾用 DataParallel 训练，会带 "module." 前缀，这里自动去掉
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

model = init_model(4).eval()
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing or unexpected:
    print("[Warn] missing keys:", missing, "| unexpected keys:", unexpected)

# ===== 导出 TorchScript=====

example = torch.randn(1, 3, 300, 300).to(device)
ts = torch.jit.trace(model, example)  # 对本模型 tracing 足够；涉及控制流可改用 script
ts.save(str(ts_path))
print(f"[OK] TorchScript saved to: {ts_path}")