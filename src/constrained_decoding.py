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


def get_best_valid_token_func(
    logits: list[float],
    valid_id: set[int],
    generated_text: str,
    function_name: list[str],
    token_id_to_text: dict[int, str]
) -> int | str:
    """
    Return the valid token ID with the highest logit.

    Args:
        logits: The logits produced by the language model.
        valid_id: The set of allowed token IDs.

    Returns:
        The valid token ID with the highest logit.
    """
    prefix = generated_text.split('"name" : "', 1)[-1]
    valid_tokens = [
        token_id
        for token_id in valid_id
        if any(
            name.startswith(
                prefix + token_id_to_text.get(token_id, "")
            )
            for name in function_name
        )
    ]
    if prefix in function_name:
        return prefix
    highest_prob = max(
        valid_tokens,
        key=lambda i: logits[i] if i < len(logits)
        else float('-inf')
    )
    return highest_prob


def get_best_valid_token_param(
        logits: list[float], valid_param_id: set[int]) -> int:
    next_param_id = max(valid_param_id, key=lambda i: logits[i])
    return next_param_id


def build_json_valid_id(vocab: Any, function_name: list[str]) -> set[int]:
    """
    Build the set of token IDs containing only valid JSON characters.

    Args:
        vocab: The tokenizer vocabulary mapping tokens to IDs.

    Returns:
        The set of token IDs containing only valid JSON characters.
    """

    valid_json = set("".join(function_name))
    valid_token = set()
    for token_str, token_id in vocab.items():
        if token_str and all(
            char in valid_json for char in token_str
        ):
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
        params = ", ".join(
            f"{name}: {info.type}"
            for name, info in len_params
        )
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
    number_rule = (
        "When a parameter type is number, integer, or float, "
        "convert number words from the user's request into numeric values."
    )
    over_param = (
        "Use only the arguments defined by the selected function."
        "When multiple values are provided, use the values needed"
        "by the function's defined parameters."
    )
    rules.extend(rule)
    rules.append(definition)
    rules.append(number_rule)
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
        "The selected function should match the requested operation.",
        "",
        "Available functions:",
        available_functions,
        "",
    ]
    return "\n".join(lines)


def get_current_function(
    generated_text: str,
    functions: Any
) -> Any | None:
    """Find the function selected by the generated JSON."""
    for function in functions:
        if f'"name" : "{function.name}"' in generated_text:
            return function
    return None


def is_parameter_position(
    generated_text: str
) -> bool:
    """
    Check whether the decoder is currently expecting
    a parameter name.
    """
    if '"arguments": {' in generated_text:
        last_part = generated_text.split(
            '"arguments": {',
            1
        )[1]
        if last_part.count('"') % 2 == 0:
            if not last_part.endswith(':'):
                return True
    return False


def build_parameter_tokens(model: Small_LLM_Model,
                           parameter_type: str) -> set[int]:
    parameter_id: set[int] = set()
    if parameter_type == "number":
        token_ids = model.encode("0123456789.-")[0].tolist()
        parameter_id.update(token_ids)
    return parameter_id
