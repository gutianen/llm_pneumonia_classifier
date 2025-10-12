from jose import jwt
from app import JWT_SECRET, ALGORITHM, JWT_ISSUER, JWT_AUDIENCE
import base64
import io
from PIL import Image
from io import BytesIO

def generate_test_token(sub: str = "test-user", role: str = "admin") -> str:
    """生成测试用JWT Token（独立于其他脚本）"""
    payload = {
        "sub": sub,
        "role": role,
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "exp": 9999999999,  # 过期时间设为遥远未来
        "iat": 1620000000
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)



def image_to_base64(path="Bacteria-1.png"):
    img = Image.open(path).convert("L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


if __name__ == "__main__":
    token = generate_test_token()
    print(f'generate test token: {token}')
    img_base64 = image_to_base64()
    print(f'generate test img base64: {img_base64}')