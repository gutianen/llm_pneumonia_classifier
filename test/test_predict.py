# test_predict.py：单独测试 /predict 接口（预测核心逻辑）
import base64
import io
from PIL import Image
import torch
import sys
from pathlib import Path
# 将项目根目录（app.py 所在目录）添加到 Python 路径
project_root = Path(__file__).parent.parent  # __file__ 是当前 test_healthz.py，parent.parent 是项目根目录
sys.path.append(str(project_root))
from fastapi.testclient import TestClient
from jose import jwt
from app import app, JWT_SECRET, ALGORITHM, JWT_ISSUER, JWT_AUDIENCE, to_tensor, DEVICE

from test_public import image_to_base64, generate_test_token

client = TestClient(app)



# ---------------------- 单独测试用例 ----------------------
def test_api_predict():
    """测试：有效Token + 有效图片 → 正常返回预测结果"""
    print("=== 测试有效输入预测 ===")
    # 1. 准备测试数据
    img_b64 = image_to_base64("COVID-1.png")
    test_token = generate_test_token(sub="predict-test-user", role="predictor")

    # 2. 发送预测请求
    response = client.post(
        "/predict",
        headers={"Authorization": f"Bearer {test_token}"},
        json={"image_b64": img_b64}  # 符合PredictReq的JSON结构
    )

    # 3. 验证结果
    assert response.status_code == 200, f"预期200，实际{response.status_code}"
    data = response.json()

    # 验证返回结构
    assert "pred" in data, "返回数据缺少'pred'字段"
    assert isinstance(data["pred"], int), f"'pred'应为整数，实际{type(data['pred'])}"
    # 若接口恢复probs字段，需添加：
    # assert "probs" in data, "返回数据缺少'probs'字段"
    # assert isinstance(data["probs"], list), f"'probs'应为列表，实际{type(data['probs'])}"

    print(f"✓ 有效输入预测成功，预测类别：{data['pred']}")


# def test_predict_with_invalid_image():
#     """测试：有效Token + 无效图片（非图片base64）→ 返回400错误"""
#     print("=== 测试无效图片预测 ===")
#     # 1. 准备无效数据（非图片的base64）
#     invalid_b64 = base64.b64encode(b"this is not an image file").decode("utf-8")
#     test_token = generate_test_token()
#
#     # 2. 发送请求
#     response = client.post(
#         "/predict",
#         headers={"Authorization": f"Bearer {test_token}"},
#         json={"image_b64": invalid_b64}
#     )
#
#     # 3. 验证结果
#     assert response.status_code == 400, f"预期400，实际{response.status_code}"
#     error_detail = response.json()["detail"].lower()
#     assert "cannot identify image file" in error_detail or "invalid image" in error_detail, \
#         f"错误信息不符合预期：{error_detail}"
#     print("✓ 无效图片测试通过")
#
#
# def test_predict_without_token():
#     """测试：无Token + 有效图片 → 返回403错误"""
#     print("=== 测试无Token预测 ===")
#     # 1. 准备有效图片
#     test_img = create_test_image()
#     img_b64 = image_to_base64(test_img)
#
#     # 2. 不携带Token发送请求
#     response = client.post(
#         "/predict",
#         json={"image_b64": img_b64}
#     )
#
#     # 3. 验证结果
#     assert response.status_code == 403, f"预期403，实际{response.status_code}"
#     assert "Not authenticated" in response.json()["detail"], "错误信息不匹配"
#     print("✓ 无Token测试通过")
#
#
# def test_image_transform_correctness():
#     """单独测试图片转换函数to_tensor：确保输出符合模型输入要求"""
#     print("=== 测试图片转换逻辑 ===")
#     # 1. 生成测试图片
#     test_img = create_test_image(size=(28, 28))
#
#     # 2. 应用转换
#     tensor = to_tensor(test_img)
#
#     # 3. 验证转换结果
#     assert tensor.shape == (1, 28, 28), f"预期形状(1,28,28)，实际{tensor.shape}"  # 1通道+28x28
#     assert tensor.dtype == torch.float32, f"预期float32，实际{tensor.dtype}"
#     assert tensor.device.type == "cpu", "转换后的张量不应绑定GPU（to_tensor无device参数）"
#
#     # 验证归一化（基于接口配置的(0.1307, 0.3081)）
#     mean = 0.1307
#     std = 0.3081
#     min_expected = (0 - mean) / std  # 约-0.424
#     max_expected = (1 - mean) / std  # 约2.822
#     assert tensor.min() >= min_expected - 0.1, f"归一化后最小值异常：{tensor.min()}"
#     assert tensor.max() <= max_expected + 0.1, f"归一化后最大值异常：{tensor.max()}"
#
#     print("✓ 图片转换逻辑测试通过")


# 单独执行所有测试
if __name__ == "__main__":
    # 先验证图片转换逻辑（依赖基础）
    # test_image_transform_correctness()
    # 再测试预测接口核心功能
    test_api_predict()
    # test_predict_with_invalid_image()
    # test_predict_without_token()
    print("\n=== /predict 接口测试通过 ===")
