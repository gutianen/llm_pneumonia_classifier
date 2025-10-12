# test_whoami.py：单独测试 /whoami 接口（鉴权相关）
import sys
from pathlib import Path
# 将项目根目录（app.py 所在目录）添加到 Python 路径
project_root = Path(__file__).parent.parent  # __file__ 是当前 test_healthz.py，parent.parent 是项目根目录
sys.path.append(str(project_root))
from fastapi.testclient import TestClient
from app import app  # 导入你的FastAPI应用实例
from jose import jwt
from app import JWT_SECRET, ALGORITHM, JWT_ISSUER, JWT_AUDIENCE
from test_public import generate_test_token



client = TestClient(app)




def test_api_whoami():
    """测试：携带有效Token访问 /whoami"""
    print("=== 测试有效Token访问 /whoami ===")
    test_token = generate_test_token(sub="test-123", role="user")

    # 发送请求
    response = client.get(
        "/whoami",
        headers={"Authorization": f"Bearer {test_token}"}  # Bearer Token格式
    )

    # 验证结果
    assert response.status_code == 200, f"预期200，实际{response.status_code}"
    data = response.json()
    print(f"response.data={data}")
    assert data["sub"] == "test-123", f"预期sub为'test-123'，实际{data['sub']}"
    assert data["role"] == "user", f"预期role为'user'，实际{data['role']}"
    print("✓ 有效Token测试通过")


# def test_whoami_with_invalid_token():
#     """测试：携带无效Token访问 /whoami"""
#     print("=== 测试无效Token访问 /whoami ===")
#     invalid_token = "invalid-token-123456"  # 错误格式的Token
#
#     # 发送请求
#     response = client.get(
#         "/whoami",
#         headers={"Authorization": f"Bearer {invalid_token}"}
#     )
#
#     # 验证结果
#     assert response.status_code == 401, f"预期401，实际{response.status_code}"
#     assert "Invalid token" in response.json()["detail"], "错误信息不包含'Invalid token'"
#     print("✓ 无效Token测试通过")
#
#
# def test_whoami_without_token():
#     """测试：不携带Token访问 /whoami"""
#     print("=== 测试无Token访问 /whoami ===")
#     # 不传递Authorization头
#     response = client.get("/whoami")
#
#     # 验证结果
#     assert response.status_code == 403, f"预期403，实际{response.status_code}"
#     assert "Not authenticated" in response.json()["detail"], "错误信息不包含'Not authenticated'"
#     print("✓ 无Token测试通过")


# 单独执行所有测试
if __name__ == "__main__":
    test_api_whoami()
    # test_whoami_with_invalid_token()
    # test_whoami_without_token()
    print("\n=== /whoami 接口测试通过 ===")
