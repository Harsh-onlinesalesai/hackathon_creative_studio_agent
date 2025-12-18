from google.adk.agents import LlmAgent
from ... import config
from .prompt import POST_PROCESSING_PROMPT
from ...models import CreativeGenerationOutput


post_processing_agent_edit = LlmAgent(
    name="post_processing_agent_edit",
    model=config.GENAI_MODEL_LOW,
    instruction=POST_PROCESSING_PROMPT,
    output_schema=CreativeGenerationOutput,
)
