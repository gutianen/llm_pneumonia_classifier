# test_metrics.py：单独测试 /metrics 接口
import sys
from pathlib import Path
# 将项目根目录（app.py 所在目录）添加到 Python 路径
project_root = Path(__file__).parent.parent  # __file__ 是当前 test_healthz.py，parent.parent 是项目根目录
sys.path.append(str(project_root))
from fastapi.testclient import TestClient
from app import app  # 导入你的FastAPI应用实例
from jose import jwt
from app import JWT_SECRET, ALGORITHM, JWT_ISSUER, JWT_AUDIENCE

client = TestClient(app)


def test_api_metrics():
    """测试：访问 /metrics 接口，返回正确的Prometheus指标"""
    print("=== 测试 /metrics 接口 ===")
    # 1. 先发送一个请求（触发指标计数，避免指标为空）
    # 生成测试Token

    test_token = jwt.encode(
        {"sub": "metrics-test", "iss": JWT_ISSUER, "aud": JWT_AUDIENCE, "exp": 9999999999},
        JWT_SECRET,
        algorithm=ALGORITHM
    )
    # 发送一个/healthz请求（触发REQ和LAT指标）
    client.get("/healthz", headers={"Authorization": f"Bearer {test_token}"})

    # 2. 访问/metrics接口
    response = client.get("/metrics")

    # 3. 验证返回格式
    assert response.status_code == 200, f"预期200，实际{response.status_code}"
    # 验证媒体类型（Prometheus标准类型）
    assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8", \
        f"媒体类型错误：{response.headers['content-type']}"



    # 4. 验证指标内容（确保核心指标存在）
    metrics_text = response.text
    print(f"response.text={metrics_text}")
    required_metrics = [
        "requests_total",  # 请求计数指标
        "requests_created",  # Counter的创建时间指标
        "request_latency_seconds"  # 延迟分布指标
    ]
    for metric in required_metrics:
        assert metric in metrics_text, f"指标{metric}未在返回中找到"
        print(f"✓ 指标{metric}存在")

    # 5. 验证指标标签（如device、route）
    assert 'device="' in metrics_text, "指标缺少device标签"
    assert 'route="/healthz"' in metrics_text, "指标缺少route标签或标签值错误"

    print("\n=== /metrics 接口测试通过 ===")


# 单独执行测试
if __name__ == "__main__":
    test_api_metrics()
