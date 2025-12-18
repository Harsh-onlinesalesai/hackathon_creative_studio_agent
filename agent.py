import os
from google.adk.agents import SequentialAgent, LlmAgent
from .sub_agents.strategy.strategy_agent import creative_strategy_agent
from .sub_agents.generation.generation_agent import creative_generation_agent
from .sub_agents.validation.validation_agent import validation_agent
from .sub_agents.post_processing.post_processing_agent import post_processing_agent
from .sub_agents.post_processing_edit.post_processing_agent import post_processing_agent_edit
from .sub_agents.image_editing.image_editing_agent import image_editing_agent
from .config import GENAI_MODEL, GENAI_MODEL_LOW
from .prompt import ROOT_PROMPT
from .models import CreativeGenerationOutput

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"

ad_creative_agent = LlmAgent(
    name="ad_creative_agent",
    description="""
    1. Analyzes ad requirements and generates multiple prompts (Awareness, CTR, etc.).
    2. Generates mock images based on those prompts.
    3. In final output, provides URLs, ids and metadata of all generated creatives in a structured JSON format.
    """,
    model=GENAI_MODEL,
    instruction=ROOT_PROMPT,
    output_schema=CreativeGenerationOutput,
    sub_agents=[
        creative_strategy_agent,
        creative_generation_agent,
        post_processing_agent
    ],
)

editing_agent = SequentialAgent(
    name="editing_agent",
    description="",
    sub_agents=[
        image_editing_agent,
        post_processing_agent_edit
    ],
)

routing_agent = LlmAgent(
    name='routing_agent',
    model=GENAI_MODEL_LOW,
    description="Routes requests to the correct agent based on the usage_mode variable in the input.",
    instruction="""Parse the given input and based on the usage_mode variable route the entire input to sub agents:
    1. If usage_mode = "create", then router it to ad_creative_agent
    2. If usage_mode = "edit", then route it to editing agent
    3. If no usage_mode is specified or any other value, use ad_creative_agent as default
    Remember: You need to pass the entire input given to you to the sub agents.
    Always call only one of the 2 agents.""",
    sub_agents=[
        ad_creative_agent,
        editing_agent
    ]
)

# This is the entry point for `adk run`
# Switch between ad_creative_agent and editing_agent as needed
root_agent = routing_agent  # Changed to editing_agent for testing