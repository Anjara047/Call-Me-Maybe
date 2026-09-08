try:
    from llm_sdk import Small_LLM_Model
except (ImportError, ModuleNotFoundError, KeyboardInterrupt):
    sys.exit()
import json

def build_json_valid_id(vocab: str):
    valid_json = set(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789*_.,-+/\'!?()[]{}"' #still missing smth
    )
    valid_token = set()
    for token_str, token_id in vocab.items():
        if token_str and all (char in valid_json for char in token_str):
            valid_token.add(token_id)
    return valid_token


def load_vocabulary(model: Small_LLM_Model):
    vocab_path = model.get_path_to_tokenizer_file()
    with open(vocab_path, 'r') as file:
        token_data = json.load(file)
    raw_vocab = token_data.get("model", {}).get("vocab", {})
    return raw_vocab

def build_sytem_prompt(function):
    lines = [
        "Available function:"
    ]
    for fn in function:
        params = ",".join(
            f"{name}: {info.type}"
            for name, info in fn.parameters.items()
        )
        lines.append(f" -{fn.name}({params}):{fn.description}")
    lines.append('\nOutput valid JSON: {"name": "<fn>", "args": {<args>}}')
    return "\n".join(lines)