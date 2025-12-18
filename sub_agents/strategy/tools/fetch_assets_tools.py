import os
import json
from google.adk.tools import ToolContext

def fetch_marketplace_assets(tool_context: ToolContext, marketplace_name: str):
    """
    Fetches static reference assets (logos, frames) for a specific marketplace.
    
    Args:
        marketplace_name: The case-sensitive name (e.g., "BigBasket").
    """
    print(f"\n[Strategy] 🖼️ Fetching static assets for: {marketplace_name}...")
    
    # Construct path relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "../../../guidelines/marketplace_assets.json")
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        assets = data.get(marketplace_name, [])
        
        if not assets:
            return f"No static assets found for {marketplace_name}."
            
        return {
            "status": "success",
            "assets": assets,
            "message": f"Found {len(assets)} static assets. You MUST include these in the 'assets' list of your plan."
        }
        
    except FileNotFoundError:
        return {"status": "error", "message": "Asset database not found."}
    except Exception as e:
        return {"status": "error", "message": str(e)}