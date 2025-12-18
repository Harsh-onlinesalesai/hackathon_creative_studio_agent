from google.adk.agents import Agent
from google.genai import types
from ... import config
from .prompt import VALIDATION_PROMPT
from .tools.inspector_tool import fetch_creative_for_inspection
from .tools.validation_result_tool import record_validation_result

# We wrap the tool to ensure the Part is returned correctly to the model
# or we can simply trust the model to interpret the tool output if ADK handles it.
# For this example, we assume `fetch_creative_for_inspection` is registered.

validation_agent = Agent(
    name="validation_agent",
    # MUST use a multimodal model (Gemini 1.5 Pro/Flash or 2.0)
    model=config.GENAI_MODEL_LOW, 
    instruction=VALIDATION_PROMPT,
    tools=[fetch_creative_for_inspection, record_validation_result],
    output_key="validation_output"
)