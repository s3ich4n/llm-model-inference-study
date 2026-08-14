from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str


class GenerateResponse(BaseModel):
    generated_text: str


class BatchGenerateRequest(BaseModel):
    prompts: list[str]


class BatchGenerateResponse(BaseModel):
    generated_texts: list[str]
