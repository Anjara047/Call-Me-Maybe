"""Functions for model initialization and JSON response generation."""
import json
import threading
import sys
from typing import Any

from src.animation import loading_animation
from src.constrained_decoding import build_json_valid_id
from src.constrained_decoding import extract_only_expected
from src.constrained_decoding import get_best_valid_token
from src.constrained_decoding import load_vocabulary
from src.constrained_decoding import get_state, get_valid_ids

try:
    from llm_sdk import Small_LLM_Model     # type: ignore
except (ImportError, ModuleNotFoundError, KeyboardInterrupt):
    print("🚫 Program stopped by the user, llm could be missing")
    sys.exit()


def initialize_model(model_name: str) -> tuple[Any, set[int]]:
    """
    Initialize the language model and valid token IDs.
    
    args:
        model_name: the model from the LLM

    return:
        all the valid token
    """
    print(f"🔥 Loading model: {model_name}")

    try:
        model = Small_LLM_Model(model_name=model_name)
    except OSError:
        print(f"Model: {model_name} not found or failed to download")
        print("This is the most probably due to unsufficient Memory")
        sys.exit()
    vocab = load_vocabulary(model)
    valid_id = build_json_valid_id(vocab)
    return model, valid_id, vocab


def generate_response(
    model: Any,
    valid_id: set[int],
    vocab: dict[str, int],
    functions: Any,
    system: str,
    user_prompt: str
):
    """
    Generate a JSON response for one user prompt.
    
    args:
        model: the model from the LLM
        valid_id: the selected valid id in vocabulary
        system: all the function available from the function definiton
        user_prompt: prompt that will be generated

    return:
        The result that was generated from the LLM
    """
    all_prompt = f"Assistant: {system}\nUser prompt: {user_prompt}"
    input_ids = model.encode(all_prompt)
    generated_ids = input_ids[0].tolist()
    all_generated_id = []
    excepted_json = None
    parsed = None
    all_generated_id.extend(
        model.encode('{"name" : "')[0].tolist()
    )
    stop_event = threading.Event()
    animation_thread = threading.Thread(
        target=loading_animation,
        args=(stop_event,)
    )
    animation_thread.start()
    try:
        while not excepted_json:
            generated_text = model.decode(all_generated_id)
            state = get_state(generated_text, functions)
            #print("\nSTATE:", repr(generated_text))
            logits = model.get_logits_from_input_ids(generated_ids + all_generated_id)
            valid_state_id = get_valid_ids(
                state,
                generated_text,
                vocab,
                functions
            )
            if not valid_state_id:
                break
            next_id = get_best_valid_token(logits,valid_state_id)
            if len(all_generated_id) > 200:
                break
            all_generated_id.append(next_id)
            generated_text = model.decode(all_generated_id)
            excepted_json = extract_only_expected(generated_text)
            if excepted_json:
                try:
                    parsed = json.loads(excepted_json)
                    break
                except json.JSONDecodeError:
                    excepted_json = None
        if parsed is None:
            print("\n⚠️ Could not generate valid", end="")
            print(" JSON within the token limit")
    finally:
        stop_event.set()
    animation_thread.join()
    return parsed
