"""Functions for model initialization and JSON response generation."""
import json
from typing import Any, cast
import threading
import sys
from pydantic import BaseModel
from src.models.pydantic_model import FunctionModel, BuildParameterModel
from src.animation import loading_animation
from src.constrained_decoding import (
    build_json_valid_id,
    extract_only_expected,
    get_best_valid_token_func,
    load_vocabulary,
    prompt_for_param,
)

try:
    from llm_sdk import Small_LLM_Model  # type: ignore
except (ImportError, ModuleNotFoundError, KeyboardInterrupt):
    print("🚫 Program stopped by the user, so the LLM is now missing")
    sys.exit()


def initialize_model(
    model_name: str,
    function_name: list[str],
) -> tuple[Any, set[int], dict[int, str]]:
    """
    Initialize the language model and valid token IDs.

    Args:
        model_name: The model to use
        function_name: all the available function to be choosen

    Return:
        model: The model that will be used
        valid_id: all the valid_id that matches the constrained
        token_id_to_text: the text from the model
    """
    print(f"🔥 Model to Use: {model_name}")
    try:
        model = Small_LLM_Model(model_name=model_name)
    except OSError:
        print(f"Model: {model_name} not found or failed to download")
        print("This is most probably due to insufficient memory")
        sys.exit()
    vocab = load_vocabulary(model)
    token_id_to_text: dict[int, str] = {
        token_id: model.decode([token_id])
        for token_str, token_id in vocab.items()
    }
    valid_id = build_json_valid_id(vocab, function_name)
    return model, valid_id, token_id_to_text


def param_generation(
    model: Any,
    generated_ids: list[int],
    function: FunctionModel,
    prompt: str,
    token_id_to_text: dict[int, str],
    param_model: type[BaseModel],
    valid_param_id: set[int],
) -> dict[str, Any] | None:
    """
    Extract and validate parameter values from a user request.

    Args:
        model_name: The model to use
        generated_ids: all the available id to generate
        function: the appropriate schema
        prompt: the user's request
        token_id_to_text: the text from the model
        param_model: a model that will be valid to get
        valid_param_id: all the vocab valid for the parameter

    Return:
        the appropriate json format
    """
    parameters = function.parameters
    description = "\n".join(
        f"- {name}: {param.type}"
        for name, param in parameters.items()
    )
    param_prompt = prompt_for_param(prompt, function, description)
    input_ids = model.encode(param_prompt)[0].tolist()
    generated: list[int] = []
    for _ in range(50):
        logits = model.get_logits_from_input_ids(input_ids + generated)
        next_id = max(range(len(logits)), key=lambda i: logits[i])
        token = token_id_to_text.get(next_id, "")
        if not token:
            break
        generated.append(next_id)
        param_text = model.decode(generated).strip()
        if not param_text:
            continue
        try:
            result = json.loads(param_text)
            for name, parameter in parameters.items():
                if name not in result:
                    if parameter.type in ("integer", "number", "float"):
                        result[name] = 0
                    elif parameter.type in ("bool", "boolean"):
                        result[name] = False
                    else:
                        result[name] = ""
            validated = param_model.model_validate(result)
            return cast(dict[str, Any], validated.model_dump())
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return None


def generate_response(
    model: Any,
    valid_id: set[int],
    system: str,
    user_prompt: str,
    functions: Any,
    token_id_to_text: dict[int, str],
    valid_param_id: set[int]
) -> dict[str, Any] | None:
    """
    Generate a JSON response for one user prompt.

    Args:
        model_name: The model to use
        valid_id: all the valid id to be choosen for the next token
        system: the model that will help for generation
        user_prompt: the user's request
        functions: all the available function in function definition
        token_id_to_text: the text from the model
        valid_param_id: a set of the id needed for the constrained

    Return:
        the valid json schema
    """
    all_prompt = f"{system}\nUser prompt: {user_prompt}\nAssistant: "
    input_ids = model.encode(all_prompt)[0].tolist()
    all_generated_id = model.encode('{"name" : "')[0].tolist()
    stop_event = threading.Event()
    animation_thread = threading.Thread(
        target=loading_animation,
        args=(stop_event,),
    )
    animation_thread.start()
    parsed = None
    selected_function: str | None = None
    generated_text = model.decode(all_generated_id)
    try:
        while not parsed:
            logits = model.get_logits_from_input_ids(
                input_ids + all_generated_id
            )
            next_id = get_best_valid_token_func(
                logits,
                valid_id,
                generated_text,
                [fn.name for fn in functions],
                token_id_to_text,
            )
            if isinstance(next_id, str):
                selected_function = next_id
                break
            all_generated_id.append(next_id)
            generated_text = model.decode(all_generated_id)
            expected_json = extract_only_expected(generated_text)
            if expected_json:
                try:
                    parsed = json.loads(expected_json)
                except json.JSONDecodeError:
                    pass
        selected_model = next(
            (
                fn for fn in functions
                if fn.name == selected_function
            ),
            None,
        )
        if selected_model is None:
            return parsed
        param_model = BuildParameterModel(selected_model)
        params = param_generation(
            model,
            input_ids,
            selected_model,
            user_prompt,
            token_id_to_text,
            param_model,
            valid_param_id
        )
        if params is None:
            return None
        return {
            "name": selected_function,
            "parameters": params,
        }
    finally:
        stop_event.set()
        animation_thread.join()
