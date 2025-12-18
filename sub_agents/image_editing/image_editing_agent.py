from google.adk.agents import Agent
from ... import config
from .prompt import IMAGE_EDITING_PROMPT
from .tools.inpainting_tool import edit_image_with_inpainting

image_editing_agent = Agent(
    name="image_editing_agent",
    model=config.GENAI_MODEL,  # Use reasoning model to call tools, not image model
    description="Performs AI-powered image editing using inpainting with mask-based editing and optional reference image guidance.",
    instruction=IMAGE_EDITING_PROMPT,
    tools=[edit_image_with_inpainting],
    output_key="image_editing_output"
)
