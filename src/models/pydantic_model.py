try:
    from pydantic import BaseModel
except (ImportError,ModuleNotFoundError):
    print("Pydantic is still missing")

class ReturnType(BaseModel):
    type: str

class Parameter(BaseModel):
    type: str

class FunctionModel(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: ReturnType

class PromptModel(BaseModel):
    prompt: str