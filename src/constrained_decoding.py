"""Manage constrained decoding and function selection."""
import sys
from typing import Any, cast
try:
    from llm_sdk import Small_LLM_Model     # type: ignore
except (ImportError, ModuleNotFoundError, KeyboardInterrupt):
    print("\n\n🚫 Since you interrupt the", end="")
    print(" program, so there is nothing to do with\n")
    sys.exit()
import json


def extract_only_expected(text: str) -> str | None:
    """
    Extract the first complete JSON object from the text.

    Args:
        text: The text containing the JSON object.

    Returns:
        The extracted JSON object, or None if no complete object is found.
    """
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


def get_best_valid_token(logits: list[float], valid_id: set[int]) -> int:
    """
    Return the valid token ID with the highest logit.

    Args:
        logits: The logits produced by the language model.
        valid_id: The set of allowed token IDs.

    Returns:
        The valid token ID with the highest logit.
    """
    highest_logits = max(
        valid_id,
        key=lambda i: logits[i] if i < len(logits) else float('-inf')
    )
    return highest_logits


def get_valid_ids(
    state: str,
    generated_text: str,
    vocab: dict[str, int],
    functions: Any
) -> set[int]:

    valid_ids = set()

    if state == "function_name":
        prefix = '{"name" : "'

        if not generated_text.startswith(prefix):
            return valid_ids

        generated_name = generated_text[len(prefix):]

        for fn in functions:
            if generated_name == fn.name + '"':
                for token, token_id in vocab.items():
                    if token == ',':
                        valid_ids.add(token_id)
                return valid_ids

        for fn in functions:
            function_name = fn.name
            if function_name.startswith(generated_name):
                remaining = function_name[len(generated_name):]
                if remaining:
                    for token, token_id in vocab.items():
                        if token and remaining.startswith(token):
                            valid_ids.add(token_id)
                else:
                    for token, token_id in vocab.items():
                        if token == '"':
                            valid_ids.add(token_id)

        return valid_ids

    selected_function = None
    for fn in functions:
        if ('"name" : "' + fn.name + '"') in generated_text:
            selected_function = fn
            break

    if selected_function is None:
        return valid_ids

    arguments_open = ', "parameters":{'

    if arguments_open not in generated_text:

        if generated_text.endswith(','):
            for token, token_id in vocab.items():
                if token == 'Ġ':
                    valid_ids.add(token_id)
            return valid_ids

        if generated_text.endswith(', '):
            for token, token_id in vocab.items():
                if token == '"':
                    valid_ids.add(token_id)
            return valid_ids

        if generated_text.endswith(', "'):
            remaining = 'parameters'
            for token, token_id in vocab.items():
                if token and remaining.startswith(token):
                    valid_ids.add(token_id)
            return valid_ids

        if generated_text.endswith(', "parameters'):
            for token, token_id in vocab.items():
                if token == '"':
                    valid_ids.add(token_id)
            return valid_ids

        if generated_text.endswith(', "parameters"'):
            for token, token_id in vocab.items():
                if token == ':':
                    valid_ids.add(token_id)
            return valid_ids

        if generated_text.endswith(', "parameters":'):
            for token, token_id in vocab.items():
                if token == '{':
                    valid_ids.add(token_id)
            return valid_ids

    arguments_start = generated_text.find(arguments_open)

    if arguments_start == -1:
        return valid_ids

    argument_text = generated_text[arguments_start + len(arguments_open):]

    if argument_text.endswith(', '):
        for token, token_id in vocab.items():
            if token == '"':
                valid_ids.add(token_id)
        return valid_ids

    if argument_text.endswith(','):
        for token, token_id in vocab.items():
            if token == 'Ġ':
                valid_ids.add(token_id)
        return valid_ids

    if argument_text.endswith('}}'):
        return valid_ids

    if argument_text.endswith('}'):
        for token, token_id in vocab.items():
            if token == '}':
                valid_ids.add(token_id)
        return valid_ids

    parts = []
    depth = 0
    in_string = False
    current = ""
    for ch in argument_text:
        if ch == '"':
            in_string = not in_string
            current += ch
        elif not in_string and ch == ',':
            parts.append(current)
            current = ""
        else:
            current += ch
    parts.append(current)

    last_part = parts[-1]
    param_types = {
        name: info.type for name, info in selected_function.parameters.items()
    }
    used_names = set()
    for part in parts[:-1]:
        if '"' in part:
            used_names.add(part.split('"')[1])

    if last_part == "":
        for token, token_id in vocab.items():
            if token == '"':
                valid_ids.add(token_id)
        return valid_ids

    if ':' not in last_part:
        if last_part.count('"') % 2 == 1:
            name_prefix = last_part.lstrip()[1:]
            for argument_name in param_types:
                if argument_name in used_names:
                    continue
                if argument_name.startswith(name_prefix):
                    remaining = argument_name[len(name_prefix):]
                    if remaining == "":
                        for token, token_id in vocab.items():
                            if token == '"':
                                valid_ids.add(token_id)
                    else:
                        for token, token_id in vocab.items():
                            if token and remaining.startswith(token):
                                valid_ids.add(token_id)
            return valid_ids

        for token, token_id in vocab.items():
            if token == ':':
                valid_ids.add(token_id)
        return valid_ids

    key, remainder = last_part.split(':', 1)
    key = key.strip('"').strip()
    param_type = param_types.get(key, "string")

    if remainder == "":
        if param_type in ("number", "integer", "float"):
            for token, token_id in vocab.items():
                if token and token[0] in '0123456789-':
                    valid_ids.add(token_id)
        else:
            for token, token_id in vocab.items():
                if token == '"':
                    valid_ids.add(token_id)
        return valid_ids

    if param_type in ("number", "integer", "float"):
        for token, token_id in vocab.items():
            if token and all(c in '0123456789.' for c in token):
                valid_ids.add(token_id)
        for token, token_id in vocab.items():
            if token == ',':
                valid_ids.add(token_id)
        for token, token_id in vocab.items():
            if token == '}':
                valid_ids.add(token_id)
        return valid_ids

    if remainder.count('"') % 2 == 1:
        safe_content = set(
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '0123456789*_.,-+/!?()[] :ĠĊĉ \n\t'
        )
        for token, token_id in vocab.items():
            if token and all(c in safe_content for c in token):
                valid_ids.add(token_id)
        for token, token_id in vocab.items():
            if token == '"':
                valid_ids.add(token_id)
        return valid_ids

    for token, token_id in vocab.items():
        if token == ',':
            valid_ids.add(token_id)
    for token, token_id in vocab.items():
        if token == '}':
            valid_ids.add(token_id)
    return valid_ids


def get_state(text: str, functions: Any) -> str:
    prefix = '{"name" : "'
    if not text.startswith(prefix):
        return "function_name"
    generated_name = text[len(prefix):]
    if '"' not in generated_name:
        return "function_name"
    name, rest = generated_name.split('"', 1)
    if not any(name == fn.name for fn in functions):
        return "function_name"
    if rest == "":
        return "function_name"
    if rest == " ,":
        return "arguments_key"
    return "arguments_key"

def build_json_valid_id(vocab: Any) -> set[int]:
    """
    Build the set of token IDs containing only valid JSON characters.

    Args:
        vocab: The tokenizer vocabulary mapping tokens to IDs.

    Returns:
        The set of token IDs containing only valid JSON characters.
    """
    valid_json = set(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789*_.,-+/\'!?()[]{}":ĠĊĉ \n\t'
    )
    valid_token = set()
    for token_str, token_id in vocab.items():
        if token_str and all(char in valid_json for char in token_str):
            valid_token.add(token_id)
    return valid_token


def load_vocabulary(model: Small_LLM_Model) -> dict[str, int]:
    """
    Load the tokenizer vocabulary from the model.

    Args:
        model: The language model containing the tokenizer information.

    Returns:
        A dictionary mapping token strings to token IDs.
    """
    vocab_path = model.get_path_to_tokenizer_file()
    with open(vocab_path, 'r') as file:
        token_data = json.load(file)
    raw_vocab = token_data.get("model", {}).get("vocab", {})
    return cast(dict[str, int], raw_vocab)



def choose_function(functions: Any) -> str:
    """
    Format the available functions for the system prompt.

    Args:
        functions: The functions available for selection.

    Returns:
        A formatted string describing the available functions.
    """
    func = []
    for fn in functions:
        len_params = list(fn.parameters.items())
        params = ", ".join(f"{name}: {info.type}" for name, info in len_params)
        func.append(f"- {fn.name}({params}): {fn.description}")
    return "\n".join(func)


def build_rules() -> list[str]:
    """Build all the system rules for the llm."""
    rules: list[str] = []

    rule = (
        "Select the function that best matches the user's intent.",
        "Use the available function names and descriptions."
    )
    over_param = (
        "Use only the parameters defined by the selected function."
        "When multiple values are provided, use the values needed"
        "by the function's defined parameters."
    )
    rules.extend(rule)
    rules.append(over_param)
    return rules


def build_system_prompt(functions: Any) -> str:
    """
    Build the system prompt used to select a matching function.

    Args:
        functions: The functions available for selection.

    Returns:
        The system prompt containing the function selection rules.
    """
    available_functions = choose_function(functions)
    rules = build_rules()
    lines = [
        *rules,
        "The selected function must match the requested operation.",
        "",
        "Available functions:",
        available_functions
    ]
    return "\n".join(lines)