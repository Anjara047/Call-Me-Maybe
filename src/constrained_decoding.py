import sys
try:
    from llm_sdk import Small_LLM_Model
except (ImportError, ModuleNotFoundError, KeyboardInterrupt):
    print("\n\n🚫 Since you interrupt the program, so there is nothing to do with\n")
    sys.exit()
import json

def extract_only_expected(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    bracket = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            bracket = bracket + 1
        if text[i] == "}":
            bracket = bracket - 1
        if bracket == 0:
            return text[start: i + 1]
    return None



def get_best_valid_token(logits, valid_id) -> int:
    highest_prob = max(
        valid_id,
        key=lambda i: logits[i] if i < len(logits) else float('-inf')
    )
    return highest_prob

def build_json_valid_id(vocab: str) -> set[int]:
    valid_json = set(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789*_.,-+/\'!?()[]{}":ĠĊĉ' #still missing smth
    )
    valid_token = set()
    for token_str, token_id in vocab.items():
        if token_str and all (char in valid_json for char in token_str):
            valid_token.add(token_id)
    return valid_token


def load_vocabulary(model: Small_LLM_Model) -> dict[str, int]:
    vocab_path = model.get_path_to_tokenizer_file()
    with open(vocab_path, 'r') as file:
        token_data = json.load(file)
    raw_vocab = token_data.get("model", {}).get("vocab", {})
    return raw_vocab


#def choose_function(function) -> str:
#    func = []
#    for fn in function:
#        if fn.function == {info.type}:
#            for name, info in fn.parameters.items:
#                func.append(f" -{fn.name}({params}):{fn.description}")
#        else:
#            return None
#
#def build_system_prompt(function) -> str:
#    lines = [
#        "Strict system rule: use only a matching function from from the list bellow.",
#        "if no function matching the user's intent(even if types match), set name: \"None\"",
#        choose_function
#        "Never use an unrealted function for different task.",
#        "",
#        "Available function:"
#    ]
#    for fn in function:
#        params = ",".join(
#            f"{name}: {info.type}"
#            for name, info in fn.parameters.items()
#        )
#        lines.append(f" -{fn.name}({params}):{fn.description}")
#    lines.append('\nOutput valid JSON: {"name": "<fn>", "args": {<args>}}')
#    return "\n".join(lines)

def choose_function(functions) -> str:
    func = []
    for fn in functions:
        len_params = list(fn.parameters.items())
        params = ", ".join(f"{name}: {info.type}" for name, info in len_params)
        func.append(f"- {fn.name}({params}): {fn.description}")
    return "\n".join(func)

def build_system_prompt(functions) -> str:
    available_functions = choose_function(functions)
    lines = [
        "Strict system rule: use only a matching function from the list below.",
        "If no function matches the user's intent (even if types match), set name: 'None'.",
        "Never use an unrelated function for a different task.",
        "",
        "Available functions:",
        available_functions,
        "",
        'Output valid JSON: {"name": "<fn>", "args": {<args>}}'
    ]
    return "\n".join(lines)
