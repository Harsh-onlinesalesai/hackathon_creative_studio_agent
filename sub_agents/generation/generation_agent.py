from google.adk.agents import Agent
from ... import config
from .prompt import GENERATION_PROMPT
from .tools.nano_banana_tool import generate_ad_creative

creative_generation_agent = Agent(
    name="creative_generation_agent",
    model=config.GENAI_MODEL,
    description="Executes the image generation tools based on the plan.",
    instruction=GENERATION_PROMPT,
    tools=[generate_ad_creative],
    output_key="generation_output"
)