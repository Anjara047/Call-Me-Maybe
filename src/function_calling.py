"""Functions for model initialization and JSON response generation."""
import json
import threading
import sys
from typing import Any
from pydantic import BaseModel
from src.models.pydantic_model import FunctionModel, BuildParameterModel
from src.animation import loading_animation
from src.constrained_decoding import (
    build_json_valid_id,
    extract_only_expected,
    get_best_valid_token_func,
    get_best_valid_token_param,
    load_vocabulary,
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
    """Initialize the language model and valid token IDs."""
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
    """Extract parameter values from the user's request."""

    parameters = function.parameters
    description = "\n".join(
        f"{name}: {param.type}"
        for name, param in parameters.items()
    )

    param_prompt = (
        f"Request: {prompt}\n"
        f"Parameters:\n{description}\n"
        "Values:\n"
    )

    input_ids = model.encode(param_prompt)[0].tolist()
    prompt_ids = set(model.encode(prompt)[0].tolist())
    generated: list[int] = []
    values: list[str] = []

    for parameter in parameters.values():
        value = ""

        while True:
            logits = model.get_logits_from_input_ids(
                input_ids + generated
            )
            next_id = get_best_valid_token_param(
                logits,
                valid_param_id,
                parameter.type,
                token_id_to_text,
                prompt_ids,
            )
            token = token_id_to_text.get(next_id, "")

            if "\n" in token:
                break
            generated.append(next_id)
            value += token

        values.append(value.strip())

    result = dict(zip(parameters, values))

    try:
        return param_model.model_validate(result).model_dump()
    except Exception:
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
    """Generate a JSON response for one user prompt."""
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
            #if len(all_generated_id) > 100:
            #    break
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
        print("SELECTED FUNCTION:", selected_function)
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
