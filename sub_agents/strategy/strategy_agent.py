from google.adk.agents import Agent
from ... import config
from .prompt import STRATEGY_PROMPT
from .tools.set_plan_tool import set_creative_plan
from .tools.fetch_guidelines_tool import fetch_marketplace_guidelines
# Import new tool
from .tools.fetch_assets_tools import fetch_marketplace_assets

creative_strategy_agent = Agent(
    name="creative_strategy_agent",
    model=config.GENAI_MODEL,
    instruction=STRATEGY_PROMPT,
    # Add to list
    tools=[
        fetch_marketplace_guidelines, 
        fetch_marketplace_assets, 
        set_creative_plan
    ], 
    output_key="strategy_output"
)