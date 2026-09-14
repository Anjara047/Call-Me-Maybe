import json
from src.models.pydantic_model import FunctionModel, PromptModel
import sys
try:
    from pydantic import ValidationError
except ModuleNotFoundError:
    sys.exit()

def load_function_definition(path: str) -> list[FunctionModel] | None:
    try:
        with open(path, 'r') as file:
            content = file.read()
        if not content.strip():
            print("No function definition found, It cannot be empty")
            return None
        function_def = json.loads(content)
        return[FunctionModel(**item) for item in function_def]
    except (FileNotFoundError, PermissionError):
        print("Error in opening file or unexistant file")
        print("It might be denied fom opening the  file, by means it was not given the permission")
        return None
    except json.JSONDecodeError:
        print("The file is not in json format")
        return None
    except ValidationError:
        print("Invalid function definition due to its wrong format")
        return None
    except RuntimeError:
        print("An error was occured in the function definition")
        return None

def load_prompt(path: str) -> list[PromptModel] | None:
    try:
        with open(path, 'r') as file:
            content = file.read()
        if not content.strip():
            print("No prompt found, It cannot be empty, Provide atleast one prompt")
            return None
        prompt = json.loads(content)
        return[PromptModel(**item) for item in prompt]
    except (FileNotFoundError, PermissionError):
        print("Error in opening file or unexistant file")
        return None
    except json.JSONDecodeError:
        print("The file is not in json format")
        return None
    except ValidationError:
        print("Invalid prompt definition")
        return None
    except RuntimeError:
        print("An error was occured in the function definition")
        return None
