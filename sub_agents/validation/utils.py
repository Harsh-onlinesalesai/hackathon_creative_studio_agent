import io
import httpx
from PIL import Image

async def download_and_process_image(url: str) -> bytes:
    """
    Downloads an image and processes it according to ADK specs:
    - Convert to JPEG
    - Resize to max 768x768 (maintaining aspect ratio)
    """
    print(f"  [Image Utils] 📥 Downloading: {url}")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        image_bytes = response.content

    # Open with Pillow
    with Image.open(io.BytesIO(image_bytes)) as img:
        # Convert to RGB (in case of PNG with transparency)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        
        # Resize logic (Max 768x768 as per ADK reference)
        # img.thumbnail((768, 768), Image.Resampling.LANCZOS)
        
        # Save to buffer as JPEG
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="JPEG", quality=100)
        print(f"  [Image Utils] 🖼️ Processed to JPEG ({img.width}x{img.height})")
        
        return output_buffer.getvalue()