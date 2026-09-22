"""Define Pydantic models for function calling."""
import sys
from typing import Any
try:
    from pydantic import BaseModel, ConfigDict
except (ImportError, ModuleNotFoundError):
    print("💡 Please run the make install to ensure", end="")
    print(" all the dependencies are available because", end="")
    print(" this project must be with all of them")
    sys.exit()

config = ConfigDict(extra='forbid')


class ReturnType(BaseModel):
    """
    Define the return type of a function.

    Attributes:
        type: The return type as a string.
    """

    model_config = config
    type: str


class Parameter(BaseModel):
    """
    Define the function parameter.

    Attributes:
        type: The return type as a string.
    """

    model_config = config
    type: str


class FunctionModel(BaseModel):
    """
    Define the structure of a function.

    Attributes:
        name: The name of the function.
        description: A description of the function.
        parameters: A dictionary mapping parameter names to their definitions.
        returns: The return type of the function.
    """

    model_config = config
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: ReturnType


class PromptModel(BaseModel):
    """
    Define the structure of a user prompt.

    Attributes:
        prompt: The user's prompt as a string.
    """

    model_config = config
    prompt: str


class OutputModel(BaseModel):
    """
    Define the structure of the output.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]
