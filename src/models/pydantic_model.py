import sys
try:
    from pydantic import BaseModel
except (ImportError,ModuleNotFoundError):
    print("💡 Please run the make install to ensure all the dependencies are available because this project must be with all of them")
    sys.exit()

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