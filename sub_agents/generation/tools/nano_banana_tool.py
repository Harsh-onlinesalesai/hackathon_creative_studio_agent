import io
import uuid
import json
import mimetypes
import httpx
import os
from pathlib import Path
from collections import Counter
# Added ImageDraw for masking
from PIL import Image, ImageOps, ImageDraw
from google import genai
from google.genai import types
from google.adk.tools import ToolContext
from .... import config
from ....config import GENAI_MODEL_LOW

from pydantic import BaseModel, Field

class AltText(BaseModel):
    text: str = Field(..., description="The generated alt text for the image.")

# Initialize Client
client = genai.Client(vertexai=True,api_key=config.GOOGLE_API_KEY)

async def download_image_as_bytes(url: str):
    """Helper to fetch image data from a URL."""
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type")
        if not content_type:
            content_type = mimetypes.guess_type(url)[0] or "image/png"
        return response.content, content_type


# --- Helper: Generate Alt Text ---
async def generate_alt_text_from_image(image_bytes: bytes, purpose: str) -> str:
    """
    Uses Gemini to look at the generated image bytes and create a descriptive Alt Text.
    """
    print(f"  [Vision] 👁️ Generating Alt Text for {purpose}...")
    try:
        # We use a fast model for description (e.g., gemini-2.0-flash)
        response = await client.aio.models.generate_content(
            model=GENAI_MODEL_LOW,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                types.Part.from_text(text="Generate a concise, accessibility-friendly Text (max 2 sentences) for this ad creative. This text will be used in meta tags along with the ad. Reminder that this is not alt text")
            ],
            config={
                "response_mime_type": "application/json",
                "response_json_schema": AltText.model_json_schema(),
            },
        )
        alt_text = AltText.model_validate_json(response.text)
        print(f"  [Vision] 📝 Alt Text: {alt_text}")
        return alt_text.text
    except Exception as e:
        print(f"  [Vision] ⚠️ Failed to generate alt text: {e}")
        return f"Ad creative for {purpose}"

# --- Helper: Upload ---
async def upload_to_service(image_bytes: bytes, filename: str, mime_type: str, client_id: str = config.CLIENT_ID):
    """
    Uploads image + JSON metadata to the service and saves a local copy.
    """
    try:
        # --- NEW STEP: Save Locally ---
        # Define a directory to save images (you can move this to your config)
        local_save_dir = "saved_images" 
        
        # Create directory if it doesn't exist
        os.makedirs(local_save_dir, exist_ok=True)
        
        # Construct the full path
        local_path = os.path.join(local_save_dir, filename)
        
        # Write bytes to file
        with open(local_path, "wb") as f:
            f.write(image_bytes)
            
        print(f"  Saved image locally: {local_path}")
        # ------------------------------

        # Get dimensions
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size

        # Prepare Multipart Form Data
        files = {
            'creative': (filename, image_bytes, mime_type)
        }
        
        # Serialize Metadata to JSON String
        data = {
            'type': 'IMAGE',
            'clientId': client_id,
            'height': str(height),
            'width': str(width),
            'name': filename,
            'tags[]': 'ai_generated',
        }

        print(f"  Uploading to service: {filename}")

        async with httpx.AsyncClient() as http_client:
            print(f"\n\n -- Creative Upload Service Request Object: {data} -- \n\n")
            response = await http_client.post(
                config.UPLOAD_SERVICE_URL,
                data=data,
                files=files,
                timeout=60.0
            )
            response.raise_for_status()
            
            result_json = response.json()
            if "creatives" in result_json and len(result_json["creatives"]) > 0:
                uploaded_data = result_json["creatives"][0]
                return {
                    "url": uploaded_data["url"],
                    "id": uploaded_data["id"],
                    "local_path": local_path 
                }
            else:
                raise ValueError(f"Unexpected response: {result_json}")

    except Exception as e:
        print(f"  Upload/Save failed: {e}")
        return None

# --- Helper: Save Locally ---
def save_image_locally(image_bytes: bytes, filename: str, metadata: dict):
    """
    Saves the generated image and its metadata to the local filesystem.
    """
    if not config.SAVE_LOCALLY:
        return None

    try:
        # Create directory if it doesn't exist
        save_dir = Path(config.LOCAL_SAVE_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save image file
        image_path = save_dir / filename
        with open(image_path, 'wb') as f:
            f.write(image_bytes)

        # Save metadata as JSON with same name
        metadata_filename = f"{Path(filename).stem}_metadata.json"
        metadata_path = save_dir / metadata_filename
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  💾 Saved locally: {image_path}")
        return str(image_path)

    except Exception as e:
        print(f"  ⚠️ Local save failed: {e}")
        return None


# --- NEW HELPER: Create Layout Mask ---
def create_layout_guide(target_w: int, target_h: int, aspect_ratio_str: str) -> tuple[bytes, int, int]:
    """
    Creates a visual guide image (Mask).
    Returns: (image_bytes, canvas_width, canvas_height)
    """
    print(f"  [Layout] 📐 Creating Layout Guide for {target_w}x{target_h} in {aspect_ratio_str}...")

    # 1. Parse Aspect Ratio (e.g., "16:9" -> 1.777)
    try:
        w_ratio, h_ratio = map(int, aspect_ratio_str.split(':'))
        ratio_float = w_ratio / h_ratio
    except:
        ratio_float = 1.0 # Default to square

    # 2. Calculate Canvas Size (Must contain target_w and target_h completely)
    # Strategy: Calculate dimensions if we fit by width, then if we fit by height. Use the larger one.
    
    # Attempt 1: Canvas Width = Target Width
    c1_w = target_w
    c1_h = int(c1_w / ratio_float)
    
    # Attempt 2: Canvas Height = Target Height
    c2_h = target_h
    c2_w = int(c2_h * ratio_float)
    
    # Select the canvas that covers the target in both dimensions
    if c1_h >= target_h:
        canvas_w, canvas_h = c1_w, c1_h
    else:
        canvas_w, canvas_h = c2_w, c2_h

    # 3. Create Image (Gray Background = Dead Zone)
    # Using a neutral gray (#808080) helps the model understand "ignore this".
    bg_color = (10, 10, 10) 
    active_color = (255, 255, 255) # White = Active Area
    
    img = Image.new('RGB', (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(img)

    # 4. Draw Centered Active Area (The Mask)
    left = (canvas_w - target_w) / 2
    top = (canvas_h - target_h) / 2
    right = left + target_w
    bottom = top + target_h
    
    draw.rectangle([left, top, right, bottom], fill=active_color)

    # 5. Convert to Bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    
    # Construct the full path
    local_path = os.path.join('saved_images', 'mask_image.png')
    
    # Write bytes to file
    with open(local_path, "wb") as f:
        f.write(buf.getvalue())

    return buf.getvalue(), canvas_w, canvas_h


# --- HELPER: Smart Crop using Mask Logic ---
def crop_to_target_dimensions(image_bytes: bytes, target_w: int, target_h: int, canvas_w: int = None, canvas_h: int = None) -> bytes:
    """
    Resizes the generated image to match the Canvas Mask dimensions, then cuts out the center.
    This guarantees alignment if the AI respected the mask.
    """
    if not target_w or not target_h:
        return image_bytes

    print(f"  [Post-Process] ✂️ Cropping to {target_w}x{target_h}...")
    
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # If we don't have canvas dims (legacy path), we fall back to simple resize-fill logic
            if not canvas_w or not canvas_h:
                 # Logic: Resize to fill target, then center crop
                current_ratio = img.width / img.height
                target_ratio = target_w / target_h
                if current_ratio > target_ratio:
                    new_h = target_h
                    new_w = int(new_h * current_ratio)
                else:
                    new_w = target_w
                    new_h = int(new_w / current_ratio)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                canvas_w, canvas_h = new_w, new_h # Treat resized img as canvas

            # Logic: Resize generated image to match the Mask Canvas calculated earlier
            # Note: Gemini generation might be higher/lower res than our calculated canvas, 
            # so we force resize to the canvas dims to align coordinates.
            else:
                 img = img.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)

            # Center Crop
            left = (canvas_w - target_w) / 2
            top = (canvas_h - target_h) / 2
            right = left + target_w
            bottom = top + target_h
            
            img_cropped = img.crop((left, top, right, bottom))
            
            # Save back to bytes
            output_buffer = io.BytesIO()
            img_cropped.save(output_buffer, format="PNG")
            return output_buffer.getvalue()

    except Exception as e:
        print(f"  [Post-Process] ⚠️ Crop failed: {e}. Using original.")
        return image_bytes


def get_closest_aspect_ratio(width, height):
    """
    Calculates the nearest aspect ratio string from a given width and height.
    Defaults to "1:1" on error.
    """
    supported_ratios = ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]
    
    try:
        # Validate inputs to ensure they are numbers
        if not (isinstance(width, (int, float)) and isinstance(height, (int, float))):
            return "1:1"
            
        # Avoid division by zero
        if height == 0:
            return "1:1"

        target_ratio = width / height

        # Helper function to convert "W:H" string to a float value
        def parse_ratio(ratio_str):
            w, h = map(int, ratio_str.split(':'))
            return w / h

        # Find the ratio in the list that has the smallest absolute difference 
        # compared to the target_ratio
        nearest = min(supported_ratios, key=lambda x: abs(target_ratio - parse_ratio(x)))
        
        return nearest

    except Exception:
        return "1:1"


# --- Main Tool Function ---
async def generate_ad_creative(
    tool_context: ToolContext, 
    prompt_text: str, 
    purpose: str, 
    original_user_prompt: str,
    assets: list[dict],
    image_size: str = "1K",
    target_width: int = None,
    target_height: int = None,
    client_id: str = config.CLIENT_ID,
):
    """
    Generates ad creative using Layout Masking, crops to custom size, generates alt-text, and uploads.
    """
    print(f"\n--- [GEMINI GEN] 🍌 Generating: {purpose} ---")
    print(f"\n--- Recieved Prompt: {prompt_text} ---")
    
    parts = []
    aspect_ratio = get_closest_aspect_ratio(target_width,target_height)
    is_extreme = False
    if target_width and target_height:
        target_ratio = target_width / target_height
        # Parse standard ratio
        w_r, h_r = map(int, aspect_ratio.split(':'))
        std_ratio = w_r / h_r
        if abs(target_ratio - std_ratio) / std_ratio > 0.35:
            is_extreme = True
            print(f"  [Post-Process] 🚨 Extreme Ratio detected. Using Edge-Extension strategy.")
    
    # --- 1. LAYOUT MASKING STRATEGY ---
    # We define these variables here so they can be used in post-processing scope
    calc_canvas_w = None
    calc_canvas_h = None

    # if target_width and target_height:
        # Generate the visual mask based on the container aspect ratio
        # mask_bytes, calc_canvas_w, calc_canvas_h = create_layout_guide(int(target_width), int(target_height), aspect_ratio)
        
        # Inject Mask into Prompt
        # parts.append(types.Part.from_bytes(data=mask_bytes, mime_type="image/png"))
        # parts.append(types.Part.from_text(text="Reference Image 0: LAYOUT_MASK, only create the image content inside this mask "))
        
        # # Inject Strict Instructions
        # prompt_text = (
        #     f"STRICT COMPOSITION INSTRUCTION: Look at Reference Image 0 labeled 'LAYOUT_MASK'. "
        #     f"It shows a black background with a White central box. "
        #     f"You MUST generate the ad content STRICTLY inside the White central box. "
        #     f"Do NOT place any text, logos, or products in the black border area. "
        #     f"\n\nAd Prompt: {prompt_text}"
        # )
        # print(f"  [Layout] 🎭 Mask injected. Canvas size: {calc_canvas_w}x{calc_canvas_h}")


    # --- 2. Add Standard Assets ---
    parts.append(types.Part.from_text(text=f"Instructions: {prompt_text}\n\nAdditional Reference Assets:"))
    for asset in assets:
        try:
            img_b, mime = await download_image_as_bytes(asset['url'])
            parts.append(types.Part.from_bytes(data=img_b, mime_type=mime))
            parts.append(types.Part.from_text(text=f"Role: {asset['role']}"))
        except: pass

    content = types.Content(role="user", parts=parts)
    gen_config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(aspect_ratio=aspect_ratio, image_size=image_size)
    )

    uploaded_creatives = []

    try:
        # 3. Generate Image
        response_stream = client.models.generate_content_stream(
            model=config.IMAGE_GEN_MODEL,
            contents=[content],
            config=gen_config,
        )

        for chunk in response_stream:
            if chunk.candidates and chunk.candidates[0].content:
                for part in chunk.candidates[0].content.parts:
                    if part.inline_data:
                        
                        raw_img_data = part.inline_data.data
                        
                        # --- 2. POST-PROCESS: EXTEND OR CROP ---
                        if target_width and target_height:
                            # if is_extreme:
                            #     raise("The given resolution are not supported by the Image Generation API")
                            # else:
                                # Use standard center crop for similar ratios (e.g. 1080x540 inside 3:2)
                            final_img_data = crop_to_target_dimensions( # Assume your previous simple center-crop exists
                                raw_img_data, 
                                int(target_width), 
                                int(target_height)
                            )
                        else:
                            final_img_data = raw_img_data
                        
                        mime_type = "image/png" 
                        ext = ".png"
                        filename = f"{purpose.replace(' ','_')}_{uuid.uuid4().hex[:6]}{ext}"
                        
                        # 5. Generate Alt Text
                        alt_text = await generate_alt_text_from_image(final_img_data, purpose)
                        
                        # 6. Metadata
                        meta_payload = {
                            "originalUserPrompt": original_user_prompt,
                            "generatedAltText": alt_text,
                            "marketingPurpose": purpose,
                            "genAiModel": config.IMAGE_GEN_MODEL,
                            "targetDimensions": f"{target_width}x{target_height}" if target_width else "Standard",
                            "layoutStrategy": "Masked" if calc_canvas_w else "Standard"
                        }

                        # 7. Save Locally (if enabled)
                        local_path = save_image_locally(final_img_data, filename, meta_payload)

                        # 8. Upload to Service
                        upload_result = await upload_to_service(
                            final_img_data,
                            filename,
                            mime_type,
                            client_id,
                            # meta_payload
                        )

                        if upload_result:
                            print(f"  ✅ Uploaded: {upload_result['url']}")
                            if local_path:
                                upload_result['local_path'] = local_path
                            uploaded_creatives.append(upload_result)

                            # Update context
                            prev = tool_context.state.get("uploaded_assets", [])
                            prev.append(upload_result)
                            tool_context.state["uploaded_assets"] = prev

    except Exception as e:
        return {"status": "error", "message": str(e)}

    return {"status": "success", "uploaded_creatives": uploaded_creatives}