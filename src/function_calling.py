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

try:
    from llm_sdk import Small_LLM_Model     # type: ignore
except (ImportError, ModuleNotFoundError, KeyboardInterrupt):
    print("🚫 Program stopped by the user, llm could be missing")
    sys.exit()


def initialize_model(model_name: str) -> tuple[Any, set[int]]:
    """Initialize the language model and valid token IDs."""
    print(f"🔥 Loading model: {model_name}")

    try:
        model = Small_LLM_Model(model_name=model_name)
    except OSError:
        print(f"Model: {model_name} not found or failed to download")
        print("This is the most probably due to unsufficient Memory")
        sys.exit()
    vocab = load_vocabulary(model)
    valid_id = build_json_valid_id(vocab)
    return model, valid_id


def generate_response(
    model: Any,
    valid_id: set[int],
    system: str,
    user_prompt: str
) -> dict[str, Any] | None:
    """Generate a JSON response for one user prompt."""
    all_prompt = f"{system}\nUser prompt: {user_prompt}\nAssistant: "
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
            logits = model.get_logits_from_input_ids(
                generated_ids + all_generated_id
            )
            next_id = get_best_valid_token(logits, valid_id)

            if len(all_generated_id) > 500:
                break
            all_generated_id.append(next_id)
            generated_text = model.decode(all_generated_id)
            excepted_json = extract_only_expected(generated_text)
            if excepted_json:
                try:
                    parsed = json.loads(excepted_json)
                    break
                except Exception:
                    pass
        if parsed is None:
            print("\n⚠️ Could not generate valid", end="")
            print(" JSON within the token limit")
    finally:
        stop_event.set()
    animation_thread.join()
    return parsed
