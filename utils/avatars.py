import os
import requests
import base64
from pathlib import Path
from dotenv import load_dotenv
from slugify import slugify

# Load .env file (must contain OPEN_ROUTER_KEY)
load_dotenv()


def generate_avatar(name: str) -> str:
    """
    Generate an AI-based professional headshot for the given name.
    Saves the image under ./avatars/<slugified_name>.png (or .jpg)
    and returns the absolute path.
    """
    prompt = f"""
    Professional corporate headshot, neutral background, soft studio lighting, sharp focus,
    natural look, subtle smile, business casual attire, centered composition.
    No text, no logos, no glasses glare, 3/4 view or straight-on.
    Name: {name}
    """

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPEN_ROUTER_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "google/gemini-2.5-flash-image",
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["image", "text"],
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()

    out_dir = Path("avatars")
    out_dir.mkdir(exist_ok=True)

    safe_name = slugify(name)
    out_path = out_dir / f"{safe_name}.png"

    if result.get("choices"):
        message = result["choices"][0]["message"]
        if message.get("images"):
            image = message["images"][0]
            image_url = image["image_url"]["url"]

            if image_url.startswith("data:image"):
                # Handle base64-encoded image
                header, b64data = image_url.split(",", 1)
                img_bytes = base64.b64decode(b64data)
                ext = "png" if "png" in header else "jpg"
                out_path = out_dir / f"{safe_name}.{ext}"
                with open(out_path, "wb") as f:
                    f.write(img_bytes)
            else:
                # Handle regular image URL
                img_bytes = requests.get(image_url).content
                with open(out_path, "wb") as f:
                    f.write(img_bytes)

            return str(out_path)

    raise RuntimeError("Failed to generate avatar from the API")


if __name__ == "__main__":
    avatar_path = generate_avatar("John Doe")
    print(f"✅ Avatar saved at {avatar_path}")
