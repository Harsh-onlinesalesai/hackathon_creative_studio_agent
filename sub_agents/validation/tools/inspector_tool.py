from google.genai import types
from google.adk.tools import ToolContext
from ..utils import download_and_process_image

async def fetch_creative_for_inspection(tool_context: ToolContext):
    """
    Retrieves the last generated creative, formats it as a JPEG Blob,
    and injects it into the visual context for validation.
    """
    # 1. Retrieve URL from session state
    uploaded_assets = tool_context.state.get("uploaded_assets", [])
    
    if not uploaded_assets:
        return "No assets found in session to validate."
        
    # Get the most recent upload
    last_asset = uploaded_assets[-1]
    url = last_asset['url']
    purpose = last_asset.get('purpose', 'Ad Creative')

    try:
        # 2. Process Image (Resize to 768px, JPEG)
        jpeg_bytes = await download_and_process_image(url)

        # 3. Create the Blob (As per ADK reference)
        image_blob = types.Blob(
            mime_type="image/jpeg",
            data=jpeg_bytes
        )

        # 4. Return as a Part
        # The Agent will see this return value and treat it as a multimodal input
        print(f"  [Validation] 👁️ Injecting image blob into model context...")
        
        return {
            "status": "success",
            "message": f"Loaded image for '{purpose}'. Please validate against guidelines.",
            # This 'visual_part' key is a convention; the important part is returning the object
            # or managing how the agent handles multimodal returns. 
            # In standard ADK, returning the Part directly or in a specific way is required.
            # Here we structure it so the agent can wrap it.
            "image_data": image_blob 
        }

    except Exception as e:
        return f"Error loading image: {str(e)}"