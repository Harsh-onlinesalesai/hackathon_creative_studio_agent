from google.adk.tools import ToolContext

def set_creative_plan(tool_context: ToolContext, creative_plan: list[dict]):
    """
    Saves the execution plan.
    
    Args:
        creative_plan: A list of dictionaries with this exact structure:
        [
            {
                "purpose": "...",
                "prompt_text": "...",
                "original_user_prompt": "The raw input from the user...",
                "assets": [...],
                "image_size": "...",
                 "target_width": 1024,         # The FINAL width required
                 "target_height": 120,         # The FINAL height required
                 "background_hex": "#000000"   # The solid color to request to fill outside the target area
            }
        ]
    """
    print(f"\n[Strategy] 🧠 Plan set with {len(creative_plan)} variations.")
    print(f"\n\n--- Set the following Creative Plan: {creative_plan}\n\n --- ")
    tool_context.state["creative_plan"] = creative_plan
    return "Plan saved successfully."