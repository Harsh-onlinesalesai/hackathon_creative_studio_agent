import os
from google.adk.tools import ToolContext

def fetch_marketplace_guidelines(tool_context: ToolContext, marketplace_name: str):
    """
    Reads the brand guidelines for a specific marketplace.
    
    Args:
        marketplace_name: The case-sensitive name (e.g., "BigBasket", "Amazon").
    """
    print(f"\n[Strategy] 📖 Fetching guidelines for: {marketplace_name}...")
    
    # Construct path relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up 3 levels to reach ad_creative/
    guideline_path = os.path.join(base_dir, "../../../guidelines", f"{marketplace_name}.txt")
    
    try:
        with open(guideline_path, "r", encoding="utf-8") as f:
            guidelines = f.read()
            
        # Optimize: If guidelines are huge, we might only return relevant sections,
        # but Gemini 2.0 has a huge context window, so reading the whole file is fine.
        return {
            "status": "success",
            "guidelines": guidelines,
            "message": "Guidelines loaded. You MUST strictly adhere to these rules in your prompts."
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "message": f"No guidelines found for '{marketplace_name}'. Use standard best practices."
        }