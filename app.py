# -*- coding: utf-8 -*-
import os, io, base64
from typing import List
from time import perf_counter

from fastapi import FastAPI, HTTPException, Response, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from pydantic import BaseModel
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
from prometheus_client.registry import CollectorRegistry
from pathlib import Path

# ===================== 配置（可用环境变量覆盖） =====================
APP_TITLE = os.getenv("APP_TITLE", "pneumonia-classifier-torchscript")
APP_VERSION = os.getenv("APP_VERSION", "1.0")
APP_DIR = Path(__file__).parent
MODEL_PATH = os.getenv("MODEL_PATH", str(APP_DIR / "torchscript" / "pneumonia_ts_model.pt"))

# JWT：默认 HS256；生产可切 RS256 并提供公钥
ALGORITHM = os.getenv("JWT_ALG", "HS256")            # HS256 / RS256
JWT_ISSUER = os.getenv("JWT_ISS", "admin-auth")
JWT_AUDIENCE = os.getenv("JWT_AUD", "admin-api")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")                 # HS256 时必须
JWT_PUBLIC_KEY = os.getenv("JWT_PUBLIC_KEY")         # RS256 时必须（PEM 公钥字符串）

# ===================== 设备选择 =====================
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ===================== 加载 TorchScript 模型 =====================
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"MODEL_PATH 不存在: {MODEL_PATH}")
model = torch.jit.load(MODEL_PATH, map_location=DEVICE).eval().to(DEVICE)

#  T 是 PyTorch 框架中 torchvision.transforms 模块别名
to_tensor = T.Compose([
    T.Resize(330),  # 调整大小保持比例
    T.CenterCrop(300),  # 中心裁剪300x300
    T.ToTensor(),  # 转换为张量
    # 相同归一化
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



# ===================== Prometheus 指标 =====================
custom_registry = CollectorRegistry()
REQ = Counter(
    "requests_total",  # 1. 指标名称（全局唯一标识）
    "Total inference requests",  # 2. 指标描述（说明指标含义，便于理解）
    labelnames=["route", "status", "device"],  # 3. 标签（接口路由、HTTP状态码、服务运行设备）
    registry=custom_registry  # 4. 指标注册表（管理指标的容器）
)
LAT = Histogram(
    "request_latency_seconds",  # 1. 指标名称
    "Latency of inference",  # 2. 指标描述
    labelnames=["route", "device"],  # 3. 标签（接口路由、HTTP状态码、服务运行设备）
    buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),  # 4. 桶（将 “延迟” 按指定区间（桶）分箱，统计每个区间的请求数量）
    registry=custom_registry  # 5. 指标注册表
)

# ===================== JWT 校验 =====================
security = HTTPBearer(auto_error=True)

def _get_verify_key() -> str:
    if ALGORITHM == "HS256":
        if not JWT_SECRET:
            raise RuntimeError("缺少 JWT_SECRET（HS256）")
        return JWT_SECRET
    elif ALGORITHM == "RS256":
        if not JWT_PUBLIC_KEY:
            raise RuntimeError("缺少 JWT_PUBLIC_KEY（RS256 公钥 PEM）")
        return JWT_PUBLIC_KEY
    else:
        raise RuntimeError(f"不支持算法: {ALGORITHM}")

def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    try:
        # 注意：python-jose 不支持 leeway 位置参数；这里使用默认严格校验
        claims = jwt.decode(
            token,
            _get_verify_key(),
            algorithms=[ALGORITHM],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
            options={
                "verify_signature": True,
                "verify_aud": True,
                "verify_exp": True,
                "verify_iat": True,
                "verify_nbf": True,
            },
        )
        return claims
    except JWTError as e:
        # 统一返回 401，便于前端识别鉴权失败
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

# ===================== Pydantic 模型 =====================
class PredictReq(BaseModel):
    image_b64: str  # base64 PNG/JPG

class PredictResp(BaseModel):
    pred: int
    # probs: List[float]

# ===================== 应用与中间件 =====================
app = FastAPI(title=APP_TITLE, version=APP_VERSION)

class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        device = str(DEVICE)
        start = perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            dur = perf_counter() - start
            LAT.labels(route=route_path, device=device).observe(dur)
            REQ.labels(route=route_path, status=status, device=device).inc()

app.add_middleware(PrometheusMiddleware)

# ===================== 路由 =====================
@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": str(DEVICE), "model_path": MODEL_PATH}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(custom_registry), media_type=CONTENT_TYPE_LATEST)

@app.get("/whoami")
def whoami(claims: dict = Depends(verify_jwt)):
    # 便于验证鉴权链路
    return {"sub": claims.get("sub"), "role": claims.get("role")}

@app.post("/predict", response_model=PredictResp)
def predict(req: PredictReq, claims: dict = Depends(verify_jwt)):
    try:
        img = Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
        x = to_tensor(img).unsqueeze(0).to(DEVICE)  # [1,3,300,300]
        with torch.inference_mode():
            logits = model(x)
            # probs = F.softmax(logits, dim=1)[0].detach().cpu().tolist()   #  F 是 PyTorch 框架中 torch.nn.functional 模块的常用别名
            # pred = int(torch.argmax(logits, dim=1).item())
            _, pred = torch.max(logits, 1)
        return PredictResp(pred=int(pred.item()))
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 配置端口、主机、工作进程数
    uvicorn.run(
        "app:app",  # 指定应用路径
        host="0.0.0.0",
        port=8801,       # 直接在代码中设置端口
        workers=1,       # 工作进程数
        reload=False,      # 开发环境热重载（生产环境关闭）
    )