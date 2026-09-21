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


def get_best_valid_token(
    logits: list[float],
    valid_id: set[int]
) -> int:
    """
    Return the valid token ID with the highest logit.

    Args:
        logits: The logits produced by the language model.
        valid_id: The set of allowed token IDs.

    Returns:
        The valid token ID with the highest logit.
    """
    highest_prob = max(
        valid_id,
        key=lambda i: logits[i] if i < len(logits) else float('-inf')
    )
    return highest_prob


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
    definition = (
        "Generate arguments according,"
        "to the selected function's definition"
    )
    over_param = (
        "Use only the arguments defined by the selected function."
        "When multiple values are provided, use the values needed"
        "by the function's defined parameters."
    )
    no_matches = (
        "If no available function matches the user's intent,"
        "return the name 'None' with empty arguments."
    )
    rules.extend(rule)
    rules.append(definition)
    rules.append(over_param)
    rules.append(no_matches)
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
        "The selected function should match the requested operation.",
        "",
        "Available functions:",
        available_functions,
        "",
        'Output valid JSON: {"name": "<fn>", "args": {<args>}}'
    ]
    return "\n".join(lines)
