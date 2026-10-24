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


def prompt_for_param(prompt: str, function: Any, description: str) -> str:
    """
    Build the prompt to extract the parameter from the prompt.

    Args:
        prompt: the user's request
        function: the available function
        description: description of each function

    Return:
        list of the prompt for the parameter extraction
    """
    rules_extraction = (
        "Extract function arguments from the request. Infer missing values "
        "from the function and parameter descriptions. "
    )

    rules_preserve = (
        "Preserve source inputs exactly. Choose regex patterns that match "
        "the requested characters or text. "
    )

    rules_replacement = (
        "Use exactly the requested replacement text, not text copied from "
        "the source or repeated. Use the exact replacement requested "
        "(e.g. 'NUMBERS' or '*'), and match each target character "
        "individually in the regex for each match. "
    )

    output_for_rule = (
        "Infer arguments without executing the function"
        "Do not invent values. "
        "For missing values, use 0 for numbers, false for booleans, "
        "or an empty string for strings. "
        "Return one valid JSON object with only the listed parameter names. "
        "Return JSON only.\nJSON:"
    )
    param_prompt = (
        f"Request: {prompt}\n"
        f"Function: {function.description}\n"
        f"Parameters: {description}\n"
        f"{rules_extraction}"
        f"{rules_preserve}"
        f"{rules_replacement}"
        f"{output_for_rule}"
    )
    return param_prompt


def build_rules() -> list[str]:
    """
    Build all the system rules for the LLM.

    Returns:
        list of all the prompt
    """
    rules: list[str] = []
    select_func = (
        "Select exactly one function that best matches"
        "the user's requested operation."
    )
    select_best_matches = (
        "Use the function names and descriptions"
        "to determine the user's intent."
    )
    matching = (
        "Match the requested operation,"
        "not just the parameter names or parameter types.",
        "Choose the function whose description"
        "most directly corresponds to the user's request.",
    )
    strict_rule = (
        "Do not select a function only because"
        "its parameters have matching types.",
    )
    no_execute = (
        "Do not execute the function or calculate its result.",
    )
    rules.extend(select_func)
    rules.extend(select_best_matches)
    rules.extend(matching)
    rules.extend(strict_rule)
    rules.extend(no_execute)
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
