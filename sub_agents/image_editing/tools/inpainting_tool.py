"""
Image Editing Tool using Gemini's Imagen capabilities.
This tool performs image editing with prompt and optional reference image guidance.
"""

import os
import io
import uuid
import json
import mimetypes
import httpx
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple
from google import genai
from google.genai import types
from google.adk.tools import ToolContext
from .... import config
from ...generation.tools.nano_banana_tool import upload_to_service
import re

# Initialize Gemini Client
client = genai.Client(vertexai=True, api_key=config.GOOGLE_API_KEY)


def parse_dimension_from_prompt(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse user prompt to detect explicit dimension or aspect ratio requests.

    Args:
        prompt: User's edit prompt text

    Returns:
        Tuple of (aspect_ratio, dimensions_str) where:
        - aspect_ratio: Detected aspect ratio like "16:9", "1:1", etc.
        - dimensions_str: Detected dimensions like "1920x1080", "1024x768", etc.

    Examples:
        "make it 16:9" -> ("16:9", None)
        "resize to 1920x1080" -> (None, "1920x1080")
        "change to square format" -> ("1:1", None)
        "make it portrait 9:16" -> ("9:16", None)
    """
    prompt_lower = prompt.lower()

    # Pattern for explicit aspect ratios (e.g., "16:9", "4:3", "1:1")
    aspect_pattern = r'\b(\d+):(\d+)\b'
    aspect_matches = re.findall(aspect_pattern, prompt)

    # Pattern for explicit dimensions (e.g., "1920x1080", "1024x768")
    dimension_pattern = r'\b(\d{3,4})\s*[x×]\s*(\d{3,4})\b'
    dimension_matches = re.findall(dimension_pattern, prompt)

    # Check for keyword-based aspect ratio requests
    aspect_keywords = {
        "square": "1:1",
        "portrait": "9:16",
        "landscape": "16:9",
        "widescreen": "16:9",
        "ultrawide": "21:9",
        "vertical": "9:16",
        "horizontal": "16:9"
    }

    detected_aspect = None
    detected_dimensions = None

    # Check aspect ratio patterns
    if aspect_matches:
        w, h = aspect_matches[0]
        detected_aspect = f"{w}:{h}"
        print(f"  🔍 Detected aspect ratio in prompt: {detected_aspect}")

    # Check dimension patterns
    if dimension_matches:
        w, h = dimension_matches[0]
        detected_dimensions = f"{w}x{h}"
        print(f"  🔍 Detected dimensions in prompt: {detected_dimensions}")

    # Check keyword-based aspect ratios
    if not detected_aspect:
        for keyword, ratio in aspect_keywords.items():
            if keyword in prompt_lower:
                detected_aspect = ratio
                print(f"  🔍 Detected aspect ratio from keyword '{keyword}': {detected_aspect}")
                break

    return detected_aspect, detected_dimensions


async def generate_edit_prompt_from_raw(raw_prompt: str, has_mask: bool, original_dimensions: str, preserve_dimensions: bool = True) -> str:
    """
    Use a flash model to automatically generate detailed edit prompt from simple user input.

    Args:
        raw_prompt: Simple user prompt like "make button gold" or "change background to blue"
        has_mask: Whether a mask is provided (affects prompt style)
        original_dimensions: Original image dimensions (e.g., "1024x768")
        preserve_dimensions: If True, instructs to preserve original dimensions; if False, allows dimension changes

    Returns:
        Detailed, professional edit prompt for image generation
    """
    print(f"  🤖 Generating detailed edit prompt from raw input using {config.GENAI_MODEL}...")

    dimension_instruction = ""
    if preserve_dimensions:
        dimension_instruction = """
- CRITICAL: MAINTAIN the original image composition, framing, and layout
- Keep the same element positioning and overall structure
- DO NOT mention dimensions, pixel sizes, or aspect ratios in your output prompt
- Focus ONLY on the visual changes requested by the user"""
    else:
        dimension_instruction = """
- The user has EXPLICITLY requested dimension or aspect ratio changes - honor their request
- Include the requested dimensions/aspect ratio in your output prompt"""

    if has_mask:
        system_instruction = f"""You are an expert at converting simple image editing requests into detailed, professional prompts for AI image editing.

Your task: Take the user's simple request and expand it into a detailed, professional editing prompt that preserves the user's intent while adding necessary visual details.

IMPORTANT GUIDELINES:
- Start by understanding and preserving the EXACT intent from the user's request
- Expand on visual details (colors, textures, lighting, effects, style)
- Maintain high quality and realistic appearance
- Focus on the specific region to be edited
- Be descriptive but concise (2-4 sentences)
{dimension_instruction}

Examples:

User request: "make button gold"
Your output: "Transform the button into a luxurious gold metallic button with elegant shine, rounded corners, premium look, and subtle shadow effect for depth"

User request: "change text color to red and make it bold"
Your output: "Change the text color to vibrant red (#FF0000) with bold, heavy font weight while maintaining clarity, readability, and professional appearance with proper contrast against the background"

User request: "make background gradient blue to purple in portrait format"
Your output: "Replace the background with a smooth gradient transitioning from deep blue at the top to rich purple at the bottom, maintaining portrait 9:16 aspect ratio with professional quality and even color distribution"

Now process the following user request and generate a detailed editing prompt:
"""
    else:
        system_instruction = f"""You are an expert at converting simple image regeneration requests into detailed, professional prompts for AI image generation.

Your task: Take the user's simple request and expand it into a detailed, professional generation prompt that preserves the user's intent while adding necessary visual details.

IMPORTANT GUIDELINES:
- Start by understanding and preserving the EXACT intent from the user's request
- Expand with specific visual details (colors, style, mood, lighting, composition)
- Maintain the subject/product from the original image unless user asks to change it
- Create professional, high-quality results
- Be descriptive but concise (2-4 sentences)
{dimension_instruction}

Examples:

User request: "blue gradient background"
Your output: "Regenerate the image featuring the product on a clean, professional blue gradient background transitioning from deep navy to bright sky blue, with soft lighting and minimal shadows"

User request: "pink aesthetic theme with dreamy vibe"
Your output: "Recreate the image with a soft pink aesthetic theme featuring pastel pink tones, dreamy bokeh lighting effects, ethereal atmosphere, and a cohesive color palette that maintains the product's visibility and appeal while creating a whimsical, dreamlike mood"

User request: "remove all text and logos"
Your output: "Regenerate the image by seamlessly removing all visible text and logos, reconstructing the underlying surfaces with matching textures and colors, maintaining professional lighting and realistic shadows throughout"

User request: "make it square 1:1 with neon lighting"
Your output: "Regenerate the image in square 1:1 aspect ratio featuring vibrant neon lighting effects with glowing edges, electric blue and pink neon accents, dramatic contrast, and a futuristic cyberpunk aesthetic while keeping the product as the focal point"

Now process the following user request and generate a detailed generation prompt:
"""

    try:
        response = await client.aio.models.generate_content(
            model=config.GENAI_MODEL,
            contents=f"{system_instruction}\n\nUser request: {raw_prompt}\n\nGenerate detailed edit prompt:",
        )

        generated_prompt = response.text.strip()
        print(f"  ✨ Generated prompt: {generated_prompt}")
        return generated_prompt

    except Exception as e:
        print(f"  ⚠️ Prompt generation failed, using raw prompt: {e}")
        return raw_prompt


def get_aspect_ratio_from_dimensions(width: int, height: int) -> str:
    """
    Calculate the closest supported Gemini aspect ratio from image dimensions.

    Supported ratios: 1:1, 3:4, 4:3, 9:16, 16:9, 21:9

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Aspect ratio string (e.g., "16:9", "1:1")
    """
    # Calculate actual ratio
    ratio = width / height

    # Define supported aspect ratios with their decimal values
    supported_ratios = {
        "1:1": 1.0,
        "3:4": 0.75,
        "4:3": 1.333,
        "9:16": 0.5625,
        "16:9": 1.778,
        "21:9": 2.333
    }

    # Find closest supported ratio
    closest_ratio = min(supported_ratios.items(), key=lambda x: abs(x[1] - ratio))

    print(f"  📐 Original dimensions: {width}x{height} (ratio: {ratio:.3f})")
    print(f"  📐 Using closest supported aspect ratio: {closest_ratio[0]} (ratio: {closest_ratio[1]:.3f})")

    return closest_ratio[0]


async def download_image_as_bytes(url: str):
    """Download image from URL and return bytes with mime type."""
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(url, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get("content-type")
        if not content_type:
            content_type = mimetypes.guess_type(url)[0] or "image/png"
        return response.content, content_type


def crop_to_target_dimensions(image_bytes: bytes, target_w: int, target_h: int) -> bytes:
    """
    Resizes and Center Crops the image to match exact target dimensions.
    Used when preserving original dimensions that don't match standard aspect ratios.
    """
    if not target_w or not target_h:
        return image_bytes

    print(f"  [Post-Process] ✂️ Cropping to exact dimensions: {target_w}x{target_h}...")

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            # Convert to RGB to avoid alpha channel issues
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            current_w, current_h = img.size

            # If already at target dimensions, return as-is
            if current_w == target_w and current_h == target_h:
                print(f"  [Post-Process] ✓ Already at target dimensions")
                return image_bytes

            # Calculate Aspect Ratios
            target_ratio = target_w / target_h
            current_ratio = current_w / current_h

            # Resize Logic (Fill Strategy) - resize so image fills target, then crop excess
            if current_ratio > target_ratio:
                # Image is wider than target -> Resize by Height
                new_h = target_h
                new_w = int(new_h * current_ratio)
            else:
                # Image is taller than target -> Resize by Width
                new_w = target_w
                new_h = int(new_w / current_ratio)

            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Center Crop Logic
            left = (new_w - target_w) / 2
            top = (new_h - target_h) / 2
            right = (new_w + target_w) / 2
            bottom = (new_h + target_h) / 2

            img_cropped = img_resized.crop((left, top, right, bottom))

            # Save back to bytes
            output_buffer = io.BytesIO()
            img_cropped.save(output_buffer, format='PNG', quality=95)
            output_buffer.seek(0)

            print(f"  [Post-Process] ✓ Cropped from {current_w}x{current_h} to {target_w}x{target_h}")
            return output_buffer.getvalue()

    except Exception as e:
        print(f"  [Post-Process] ⚠️ Crop failed: {e}. Returning original.")
        return image_bytes


def save_image_locally(image_bytes: bytes, filename: str, metadata: dict) -> str:
    """Save image and metadata to local directory."""
    if not config.SAVE_LOCALLY:
        return None

    try:
        save_dir = Path(config.LOCAL_SAVE_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        image_path = save_dir / filename
        with open(image_path, 'wb') as f:
            f.write(image_bytes)

        # Save metadata
        metadata_filename = f"{Path(filename).stem}_metadata.json"
        metadata_path = save_dir / metadata_filename
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  💾 Saved locally: {image_path}")
        return str(image_path)

    except Exception as e:
        print(f"  ⚠️ Local save failed: {e}")
        return None


def build_edit_prompt(edit_prompt: str, reference_image_url: Optional[str] = None) -> str:
    """
    Build enhanced editing prompt with reference image guidance.
    For visual similarity, we instruct to match style and aesthetic.
    """
    if reference_image_url:
        enhanced = f"{edit_prompt}\n\nIMPORTANT: Match the visual style, color palette, composition, lighting, texture quality, and overall aesthetic of the reference image provided. Maintain visual similarity while incorporating the requested changes."
        return enhanced
    return edit_prompt


def composite_edited_region(
    original_image_bytes: bytes,
    edited_content_bytes: bytes,
    mask_bytes: bytes
) -> bytes:
    """
    Composite the edited content back into the original image using the mask.

    Args:
        original_image_bytes: Original source image
        edited_content_bytes: Generated/edited content from Gemini
        mask_bytes: Mask image (white = edited region, black = keep original)

    Returns:
        Composited image bytes
    """
    print(f"  🎨 Compositing edited region back into original image...")

    # Load images
    original = Image.open(io.BytesIO(original_image_bytes)).convert("RGBA")
    edited = Image.open(io.BytesIO(edited_content_bytes)).convert("RGBA")
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

    # Resize edited content to match original dimensions if needed
    if edited.size != original.size:
        print(f"  ↔️  Resizing edited content from {edited.size} to {original.size}")
        edited = edited.resize(original.size, Image.Resampling.LANCZOS)

    # Resize mask to match original if needed
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.Resampling.LANCZOS)

    # Composite: use edited content where mask is white, original where black
    composited = Image.composite(edited, original, mask)

    # Convert back to RGB for saving
    composited_rgb = composited.convert("RGB")

    # Save to bytes
    output = io.BytesIO()
    composited_rgb.save(output, format='PNG', quality=95)
    output.seek(0)

    print(f"  ✅ Compositing complete - edited region blended into original")
    return output.getvalue()


async def edit_image_with_gemini(
    image_bytes: bytes,
    image_mime: str,
    mask_bytes: bytes,
    mask_mime: str,
    edit_prompt: str,
    aspect_ratio: str,
    reference_bytes: Optional[bytes] = None,
    reference_mime: Optional[str] = None,
    image_size: str = "1K"
) -> bytes:
    """
    Use Gemini to edit an image based on a mask and prompt.
    Optionally includes reference image for style guidance.
    """
    print(f"  [GEMINI EDIT] 🎨 Editing image with Gemini...")

    # Build content parts
    parts = []

    # Add editing instruction - adjust based on whether mask is provided
    if mask_bytes:
        # MASK-BASED EDITING - keep existing complex instruction
        instruction = f"""You are an expert image editor. Edit the provided image according to these instructions:

{edit_prompt}

The masked areas (shown in white in the mask image) should be edited according to the prompt.
The unmasked areas (black in mask) should remain unchanged.

Maintain high quality, realistic details, and ensure seamless blending between edited and original regions."""

        if reference_bytes:
            instruction += "\n\nA reference image is provided - match its visual style, color palette, and aesthetic."
            parts.append(types.Part.from_bytes(data=reference_bytes, mime_type=reference_mime))
            parts.append(types.Part.from_text(text="Reference image for visual style guidance."))

        parts.append(types.Part.from_text(text=instruction))
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime))
        parts.append(types.Part.from_text(text="Original image to edit."))
        parts.append(types.Part.from_bytes(data=mask_bytes, mime_type=mask_mime))
        parts.append(types.Part.from_text(text="Mask image (white = edit region, black = keep original)."))
    else:
        # PROMPT-BASED REGENERATION - match create flow structure
        instruction = f"Instructions: {edit_prompt}\n\nReference Assets:"

        parts.append(types.Part.from_text(text=instruction))

        # Add reference image first if provided
        if reference_bytes:
            parts.append(types.Part.from_bytes(data=reference_bytes, mime_type=reference_mime))
            parts.append(types.Part.from_text(text="Role: Style Reference"))

        # Add original image as reference
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime))
        parts.append(types.Part.from_text(text="Role: Original Image"))

    content = types.Content(role="user", parts=parts)

    gen_config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(aspect_ratio=aspect_ratio, image_size=image_size)
    )

    try:
        # Generate edited image
        response_stream = client.models.generate_content_stream(
            model=config.IMAGE_GEN_MODEL,
            contents=[content],
            config=gen_config,
        )

        # Extract edited image from response
        for chunk in response_stream:
            if chunk.candidates and chunk.candidates[0].content:
                for part in chunk.candidates[0].content.parts:
                    if part.inline_data:
                        print(f"  ✅ Image editing completed")
                        return part.inline_data.data

        print(response_stream)

        raise Exception("No image data returned from Gemini")

    except Exception as e:
        raise Exception(f"Gemini editing failed: {str(e)}")


async def edit_image_with_inpainting(
    tool_context: ToolContext,
    image_url: str,
    edit_prompt: str,
    mask_url: Optional[str] = None,
    reference_image_url: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    output_filename: Optional[str] = None,
    raw_mode: bool = True,
    client_id:str = config.CLIENT_ID
) -> dict:
    """
    Edit an image using Gemini's image generation with optional mask-based editing and reference image.

    THREE MODES:
    1. MASK-BASED EDITING: Provide mask_url to edit specific regions (white areas in mask)
    2. PROMPT-BASED REGENERATION: Omit mask_url to regenerate entire image based on prompt
    3. RAW MODE: Set raw_mode=True to auto-generate detailed prompts from simple user input (DEFAULT)

    Args:
        tool_context: ADK tool context for state management
        image_url: URL of the source image to edit or use as reference
        edit_prompt: Text description of how to edit/regenerate the image
                    If raw_mode=True, this can be simple like "make button gold" or "blue background"
                    If raw_mode=False, this should be detailed and professional
        mask_url: Optional URL of mask image (white = areas to edit, black = areas to keep)
                 If None, entire image will be regenerated based on prompt
        reference_image_url: Optional reference image URL for visual similarity guidance
        aspect_ratio: Output aspect ratio (e.g., "1:1", "16:9", "3:2"). If None, uses original image aspect ratio.
        output_filename: Optional custom filename for saved output
        raw_mode: If True (DEFAULT), uses flash model to auto-generate detailed prompt from simple input

    Returns:
        Dictionary containing the edited image path, metadata, and status
    """
    print(f"\n--- [IMAGE EDIT] 🖼️ Starting image editing ---")

    try:
        # Download input images
        print(f"  📥 Downloading source image from: {image_url}")
        image_bytes, image_mime = await download_image_as_bytes(image_url)

        # Detect original image dimensions
        with Image.open(io.BytesIO(image_bytes)) as img:
            original_width, original_height = img.size

        original_dimensions = f"{original_width}x{original_height}"
        print(f"  📏 Original image dimensions: {original_dimensions}")

        # Parse prompt for explicit dimension/aspect ratio requests
        detected_aspect, detected_dimensions = parse_dimension_from_prompt(edit_prompt)

        # Determine final aspect ratio and whether to preserve dimensions
        preserve_dimensions = True
        if aspect_ratio is not None:
            # User explicitly passed aspect_ratio parameter
            print(f"  📐 Using user-specified aspect ratio parameter: {aspect_ratio}")
            preserve_dimensions = False
        elif detected_aspect is not None:
            # User mentioned aspect ratio in their prompt
            aspect_ratio = detected_aspect
            print(f"  📐 Using aspect ratio detected from prompt: {aspect_ratio}")
            preserve_dimensions = False
        elif detected_dimensions is not None:
            # User mentioned specific dimensions in their prompt
            # Calculate aspect ratio from detected dimensions
            w, h = map(int, detected_dimensions.split('x'))
            aspect_ratio = get_aspect_ratio_from_dimensions(w, h)
            print(f"  📐 Using aspect ratio from detected dimensions {detected_dimensions}: {aspect_ratio}")
            preserve_dimensions = False
        else:
            # No explicit dimension request - use original aspect ratio
            aspect_ratio = get_aspect_ratio_from_dimensions(original_width, original_height)
            print(f"  📐 No dimension request found - preserving original aspect ratio: {aspect_ratio}")
            preserve_dimensions = True

        mask_bytes = None
        mask_mime = None
        if mask_url:
            print(f"  📥 Downloading mask image from: {mask_url}")
            mask_bytes, mask_mime = await download_image_as_bytes(mask_url)
            print(f"  🎭 Mode: MASK-BASED EDITING (editing specific regions)")
        else:
            print(f"  ✨ Mode: PROMPT-BASED REGENERATION (full image regeneration)")

        # Download reference image if provided
        reference_bytes = None
        reference_mime = None
        if reference_image_url:
            print(f"  📥 Downloading reference image from: {reference_image_url}")
            reference_bytes, reference_mime = await download_image_as_bytes(reference_image_url)

        # Generate detailed prompt if raw_mode is enabled
        if raw_mode:
            print(f"  📝 Raw mode enabled - original prompt: {edit_prompt}")
            print(f"  📝 Dimension preservation mode: {'PRESERVE original' if preserve_dimensions else 'CHANGE per user request'}")
            edit_prompt = await generate_edit_prompt_from_raw(
                edit_prompt,
                has_mask=mask_url is not None,
                original_dimensions=original_dimensions,
                preserve_dimensions=preserve_dimensions
            )
        else:
            print(f"  📝 Raw mode disabled - using prompt as-is: {edit_prompt}")

        # Build enhanced prompt
        final_prompt = build_edit_prompt(edit_prompt, reference_image_url)
        print(f"  📋 Final edit prompt: {edit_prompt}")

        # Call Gemini for image editing
        edited_content_bytes = await edit_image_with_gemini(
            image_bytes=image_bytes,
            image_mime=image_mime,
            mask_bytes=mask_bytes,
            mask_mime=mask_mime,
            edit_prompt=final_prompt,
            aspect_ratio=aspect_ratio,
            reference_bytes=reference_bytes,
            reference_mime=reference_mime
        )

        # Composite the edited content back into the original image (only if mask provided)
        if mask_url and mask_bytes:
            edited_image_bytes = composite_edited_region(
                original_image_bytes=image_bytes,
                edited_content_bytes=edited_content_bytes,
                mask_bytes=mask_bytes
            )
        else:
            # No mask - use generated image directly (full regeneration)
            edited_image_bytes = edited_content_bytes

        # Apply crop to exact original dimensions if preserving dimensions
        if preserve_dimensions:
            print(f"  🎯 Dimension preservation enabled - cropping to exact original dimensions")
            edited_image_bytes = crop_to_target_dimensions(
                edited_image_bytes,
                original_width,
                original_height
            )

        # Generate filename
        if not output_filename:
            output_filename = f"edited_image_{uuid.uuid4().hex[:8]}.png"

        # Prepare metadata
        metadata = {
            "edit_prompt": edit_prompt,
            "reference_image_url": reference_image_url,
            "aspect_ratio": aspect_ratio,
            "original_dimensions": original_dimensions,
            "dimensions_preserved": preserve_dimensions,
            "detected_aspect_in_prompt": detected_aspect,
            "detected_dimensions_in_prompt": detected_dimensions,
            "model": config.IMAGE_GEN_MODEL,
            "source_image_url": image_url,
            "mask_url": mask_url,
            "raw_mode": raw_mode
        }

        # Save locally
        local_path = save_image_locally(edited_image_bytes, output_filename, metadata)

        # Get image dimensions
        with Image.open(io.BytesIO(edited_image_bytes)) as img:
            width, height = img.size

        uploaded_obj = await upload_to_service(
            image_bytes=edited_image_bytes,
            filename=output_filename,
            mime_type=image_mime,
            client_id=client_id
        )

        result = {
            "success": True,
            "prompt_used": edit_prompt,
            "dimensions": f"{width}x{height}",
            "metadata": metadata,
            "upload_results": uploaded_obj,
            "message": "Image edited successfully using Gemini"
        }

        print(f"  ✓ Image editing completed. Saved to: {local_path or 'Not saved locally'}")

        # Update tool context
        prev_edits = tool_context.state.get("edited_images", [])
        prev_edits.append(result)
        tool_context.state["edited_images"] = prev_edits

        return result

    except Exception as e:
        error_msg = f"Error during image editing: {str(e)}"
        print(f"  ❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "local_path": None,
        }
