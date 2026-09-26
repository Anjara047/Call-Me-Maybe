"""Functions for model initialization and JSON response generation."""
import json
import threading
import sys
from typing import Any
from src.models.pydantic_model import FunctionModel
# from src.models.pydantic_model import PromptModel
from src.animation import loading_animation
from src.constrained_decoding import build_json_valid_id
from src.constrained_decoding import extract_only_expected
from src.constrained_decoding import get_best_valid_token_func
from src.constrained_decoding import get_best_valid_token_param
from src.constrained_decoding import load_vocabulary
# from src.constrained_decoding import get_current_function
# from src.constrained_decoding import is_parameter_position
# from src.constrained_decoding import build_parameter_tokens

try:
    from llm_sdk import Small_LLM_Model     # type: ignore
except (ImportError, ModuleNotFoundError, KeyboardInterrupt):
    print("🚫 Program stopped by the user, So our llm is now missing")
    sys.exit()


def initialize_model(
        model_name: str, function_name: list[str]) -> tuple[Any, set[int], dict[int, str]]:
    """Initialize the language model and valid token IDs."""
    print(f"🔥 Model to Use: {model_name}")

    try:
        model = Small_LLM_Model(model_name=model_name)
    except OSError:
        print(f"Model: {model_name} not found or failed to download")
        print("This is the most probably due to unsufficient Memory")
        sys.exit()
    vocab = load_vocabulary(model)
    token_id_to_text: dict[int, str] = {
        token_id: token_str
        for token_str, token_id in vocab.items()
    }
    valid_id = build_json_valid_id(vocab, function_name)
    return model, valid_id, token_id_to_text


def param_generation(
    model: Any,
    generated_ids: list[int],
    functions: FunctionModel,
    prompt: str,
    token_id_to_text: dict[int, str]
) -> dict[str, Any] | None:
    parameters = functions.parameters
    param_description = []
    get_result = {}
    for name, param in parameters.items():
        param_description.append(
            f"{name}: {param.type}"
        )
    #for token_id in model.encode(prompt)[0].tolist():
    #    print(token_id, repr(token_id_to_text.get(token_id)))
    promp_param = (
        "Extract the parameter values directly from the user's request.\n"
        "Convert number words to digits when needed.\n"
        "Do not calculate or invent values.\n"
        "Return one value per line, in parameter order.\n\n"
        f"Request for the user:\n{prompt}\n\n"
        f"Parameters:\n{'\n'.join(param_description)}"
    )
    param_input_ids = model.encode(promp_param)
    prompt_token_ids = set(model.encode(prompt)[0].tolist())
    #print("PROMPT TOKEN IDS:", prompt_token_ids)
    for token_id in model.encode(prompt)[0].tolist():
        token_text = token_id_to_text.get(token_id, "")
        #print(token_id, repr(token_text), token_text.strip().isdigit())
    param_ids = param_input_ids[0].tolist()
    all_param_ids = param_ids.copy()
    prompt_digits = set(char for char in prompt if char.isdigit())
    for name, param in parameters.items():
        generated_param_ids: list[int] = []
        # print("PARAM:", name)
        # print("TYPE:", param.type)
        # print("PROMPT:", prompt)
        prompt_ids = model.encode(prompt)[0].tolist()

        #if param.type == "number":
        #    valid_param_id = {
        #        token_id
        #        for token_id in prompt_ids
        #        if token_id_to_text.get(token_id, "").strip().isdigit()
        #    }
        #print("VALID PARAM IDS:", valid_param_id)
        while True:
            #print(model.encode(prompt)[0].tolist())
            logits = model.get_logits_from_input_ids(all_param_ids)
            valid_param_id = {
               token_id for token_id, token_text in token_id_to_text.items()
            }
            if len(generated_param_ids) > 0 and generated_param_ids[-1] != 198:
                valid_param_id.add(198)
            next_id = get_best_valid_token_param(logits, valid_param_id)
            all_param_ids.append(next_id)
            generated_param_ids.append(next_id)
            generated_text = model.decode(generated_param_ids)
            next_id = get_best_valid_token_param(logits, valid_param_id)
            if generated_text.count("\n") >= len(parameters):
                break
            if len(generated_param_ids) >= 20:
                break
            get_result[name] = generated_text
    values = generated_text.strip().split("\n")
    param_name = list(parameters.keys())
    param_result = {}
    for param in range(len(param_name)):
        if param >= len(values):
            break
        param_result[param_name[param]] = values[param]
    return param_result


def generate_response(
    model: Any,
    valid_id: set[int],
    system: str,
    user_prompt: str,
    functions: Any,
    token_id_to_text: dict[int, str],
) -> dict[str, Any]:
    """Generate a JSON response for one user prompt."""
    all_prompt = f"{system}\nUser prompt: {user_prompt}\nAssistant: "
    input_ids = model.encode(all_prompt)
    generated_ids = input_ids[0].tolist()
    all_generated_id = model.encode('{"name" : "')[0].tolist()
    stop_event = threading.Event()
    animation_thread = threading.Thread(
        target=loading_animation,
        args=(stop_event,)
    )
    animation_thread.start()
    parsed = None
    selected_function: str | None = None
    generated_text = model.decode(all_generated_id)
    try:
        while not parsed:
            logits = model.get_logits_from_input_ids(
                generated_ids + all_generated_id
            )
            next_id = get_best_valid_token_func(
                logits,
                valid_id,
                generated_text,
                [fn.name for fn in functions],
                token_id_to_text
            )
            if isinstance(next_id, str):
                selected_function = next_id
                break
            all_generated_id.append(next_id)
            generated_text = model.decode(all_generated_id)
            excepted_json = extract_only_expected(generated_text)
            if excepted_json:
                try:
                    parsed = json.loads(excepted_json)
                except Exception:
                    pass
            if len(all_generated_id) > 10:
                break
        model_selected = None
        for fn in functions:
            if fn.name == selected_function:
                model_selected = fn
        if model_selected is not None:
            params = param_generation(
                model,
                generated_ids,
                model_selected,
                user_prompt,
                token_id_to_text)
            return {"name": selected_function, "parameters": params}
    finally:
        stop_event.set()
    animation_thread.join()
    return parsed
