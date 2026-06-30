import base64
import os
from openai import OpenAI

# Load .env file
def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key not in os.environ:
                os.environ[key] = value

load_dotenv()

API_KEY = os.getenv("DMX_API_KEY")
BASE_URL = os.getenv("DMX_BASE_URL", "https://www.dmxapi.cn/v1")
IMAGE_PATH = "test_image.png"

if not API_KEY:
    print("[!] DMX_API_KEY not set. Put it in .env file or set as env var.")
    exit(1)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    if not os.path.exists(IMAGE_PATH):
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (400, 200), color=(135, 181, 158))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 350, 150], fill=(232, 145, 106))
        draw.text((120, 80), "Hello DMXAPI!", fill="white")
        img.save(IMAGE_PATH)
        print(f"[*] Created test image: {IMAGE_PATH}")

    image_b64 = encode_image(IMAGE_PATH)
    print(f"[*] Image encoded, size: {len(image_b64)} chars")

    print("[*] Calling gpt-4o via DMXAPI...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这张图片里有什么内容，用中文回答。"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "auto",
                        },
                    },
                ],
            }
        ],
        max_tokens=500,
    )

    result = response.choices[0].message.content
    print("=" * 50)
    print("Model:", response.model)
    print("Response:")
    print(result)
    print("=" * 50)
    print(f"Tokens: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}, total={response.usage.total_tokens}")


if __name__ == "__main__":
    main()
