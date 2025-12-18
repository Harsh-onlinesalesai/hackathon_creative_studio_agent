from pydantic import BaseModel, Field


class CreativeGeneration(BaseModel):
    # each generated creative with url and metadata
    url: str = Field(description="URL of the generated creative.")
    id: str = Field(description="Unique identifier for the generated creative recieved from the creative upload service.")
    metadata: dict = Field(description="Metadata associated with the generated creative.")

class CreativeGenerationOutput(BaseModel):
    # the output can have mulitple generated creatives
    creatives: list[CreativeGeneration] = Field(description="List of generated creatives with their metadata.")

