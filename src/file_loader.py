"""Provide functions for reading JSON configuration files."""
import json
from src.models.pydantic_model import FunctionModel, PromptModel
import sys
import os
try:
    from pydantic import ValidationError
except ModuleNotFoundError:
    sys.exit()


def load_function_definition(path: str) -> list[FunctionModel] | None:
    """
    Load function definitions from a JSON file.

    Args:
        path: The path to the JSON file containing the function definitions.

    Returns:
        A list of function definitions, or None if the file is invalid.
    """
    try:
        with open(path, 'r') as file:
            content = file.read()
        if not content.strip():
            print("No function definition found, It cannot be empty")
            return None
        function_def = json.loads(content)
        return [FunctionModel(**item) for item in function_def]
    except (FileNotFoundError, PermissionError):
        print("Error in opening file or unexistant file")
        print("🔏It might be denied fom opening the file,", end="")
        print(" by means it was not given the permission")
        return None
    except json.JSONDecodeError:
        print("The file that contains", end="")
        print(" the function definition is not in json format")
        return None
    except ValidationError:
        print("Invalid function definition due to its wrong format")
        print("Or the user may add extra parameter")
        return None
    except RuntimeError:
        print("An error was occured in the function definition")
        return None


def load_prompt(path: str) -> list[PromptModel] | None:
    """
    Load prompts from a JSON file.

    Args:
        path: The path to the JSON file containing the prompts.

    Returns:
        A list of prompts, or None if the file is invalid.
    """
    try:
        with open(path, 'r') as file:
            content = file.read()
        if not content.strip():
            print("No prompt found, It cannot be empty,", end="")
            print(" Provide atleast one prompt")
            return None
        prompt = json.loads(content)
        return [PromptModel(**item) for item in prompt]
    except (FileNotFoundError, PermissionError):
        print("Error in opening file or unexistant file")
        print("🔏It might be denied fom opening the file,", end="")
        print(" by means it was not given the permission")
        return None
    except json.JSONDecodeError:
        print("The file that contains the prompt is not in json format")
        return None
    except ValidationError:
        print("Invalid prompt definition or extra parameter")
        return None
    except RuntimeError:
        print("An error was occured in the function definition")
        return None


def save_results(
    path: str,
    results: list[dict]
) -> None:
    """
    Save function-calling results to a JSON file.

    Args:
        path: The path to the output JSON file.
        results: The function-calling results to save.

    Returns:
        None.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            json.dump(
                results,
                file,
                ensure_ascii=False,
                indent=2
            )
        print(f"\n╰┈➤ˎˊ˗ Result saved to : {path}")
    except PermissionError:
        print("⚠️You denied the permission from the file to save the result")
        print("So the result is not saved anywhere")
