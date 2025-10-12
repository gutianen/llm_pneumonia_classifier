# test_healthz.py：单独测试 /healthz 接口
import sys
from pathlib import Path
# 将项目根目录（app.py 所在目录）添加到 Python 路径
project_root = Path(__file__).parent.parent  # __file__ 是当前 test_healthz.py，parent.parent 是项目根目录
sys.path.append(str(project_root))
from fastapi.testclient import TestClient
from app import app  # 导入你的FastAPI应用实例

# 创建测试客户端（模拟HTTP请求）
client = TestClient(app)


def test_api_healthz():
    """单独测试 /healthz 接口：正常返回服务状态"""
    print("=== 开始测试 /healthz 接口 ===")

    # 发送GET请求
    response = client.get("/healthz")

    # 验证状态码
    assert response.status_code == 200, f"预期状态码200，实际得到{response.status_code}"
    print("✓ 状态码验证通过（200 OK）")

    # 验证返回数据结构
    data = response.json()
    required_keys = ["status", "device", "model_path"]
    for key in required_keys:
        assert key in data, f"返回数据缺少必要字段：{key}"
    print("✓ 返回数据结构验证通过")

    # 验证具体内容
    assert data["status"] == "ok", f"预期status为'ok'，实际得到{data['status']}"
    assert data["device"] in ["cpu", "cuda", "mps"], f"无效设备类型：{data['device']}"
    print(f"✓ 服务状态：{data['status']}")
    print(f"✓ 运行设备：{data['device']}")
    print(f"✓ 模型路径：{data['model_path']}")

    print("=== /healthz 接口测试全部通过 ===")


# 单独执行测试
if __name__ == "__main__":
    test_api_healthz()
