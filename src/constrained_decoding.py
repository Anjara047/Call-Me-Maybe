import sys
try:
    from llm_sdk import Small_LLM_Model
except (ImportError, ModuleNotFoundError, KeyboardInterrupt):
    sys.exit()
import json

def extract_only_expected(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    stack = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            stack = stack + 1
        if text[i] == "}":
            stack = stack - 1
        if stack == 0:
            return text[start: i + 1]
    return None



def get_best_valid_token(logits, valid_id) -> float:
    highest_prob = max(
        valid_id,
        key=lambda i: logits[i] if i < len(logits) else float('-inf')
    )
    #print("best:",highest_prob)
    return highest_prob

#def _next_valid_chars(text: str) -> set[str]:
#    """Given the JSON text generated so far, return the set of characters
#    that could legally come next (structural JSON validity only)."""
#    stripped = text.rstrip()
#    in_string = text.count('"') % 2 == 1
#
#    if in_string:
#        return set(chr(c) for c in range(32, 127)) - {'\\'}
#
#    if stripped == "":
#        return {'{'}
#
#    last = stripped[-1]
#
#    if last == '{':
#        return {'"', '}'}
#    if last == '"':
#        return {':', ',', '}'}
#    if last == ':':
#        return {'"', '{'} | set('0123456789-')
#    if last == ',':
#        return {'"'}
#    if last in '0123456789':
#        return set('0123456789.eE') | {',', '}'}
#    if last == '}':
#        return {',', '}'}
#    return set()
#
#
#def is_valid_continuation(current_text: str, addition: str) -> bool:
#    text = current_text
#    for ch in addition:
#        valid = _next_valid_chars(text)
#        if valid and ch not in valid:
#            return False
#        text += ch
#    return True
#
#
#def build_json_valid_id(vocab: dict, current_text: str) -> set[int]:
#    valid_token = set()
#    for token_str, token_id in vocab.items():
#        clean = token_str.replace('Ġ', ' ')
#        if clean and is_valid_continuation(current_text, clean):
#            valid_token.add(token_id)
#    return valid_token
def build_json_valid_id(vocab: str) -> set[int]:
    valid_json = set(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789*_.,-+/\'!?()[]{}":Ġ' #still missing smth
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

def build_system_prompt(function):
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