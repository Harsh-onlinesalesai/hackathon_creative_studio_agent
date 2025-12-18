from google.adk.agents import LlmAgent
from ... import config
from .prompt import POST_PROCESSING_PROMPT
from ...models import CreativeGenerationOutput


post_processing_agent = LlmAgent(
    name="post_processing_agent",
    model=config.GENAI_MODEL_LOW,
    instruction=POST_PROCESSING_PROMPT,
    output_schema=CreativeGenerationOutput,
)
