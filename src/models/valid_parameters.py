"""Provide functions for validating and casting parameters."""
from src.models.pydantic_model import Parameter


def cast_parameter(
    value: object,
    expected_type: str
) -> object | None:
    """
    Cast a value into its expected type.

    Args:
        value: The value of the parameter to cast.
        expected_type: The expected type from the function definition.

    Returns:
        The casted value, or None if the value cannot be cast.
    """
    if expected_type == "integer":
        if isinstance(value, (int, float, str)):
            try:
                return int(value)
            except (ValueError, TypeError) as e:
                print(e)
                return None
            return None
    if expected_type == "float":
        if isinstance(value, (int, float, str)):
            try:
                return float(value)
            except (ValueError, TypeError) as e:
                print(e)
                return None
            return None
    if expected_type == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            try:
                if "." in value:
                    return float(value) + 0.0
                return int(value)
            except ValueError:
                return None
        return None
    if expected_type == "string":
        return str(value)
    if expected_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            value = value.lower()
            if value == "true":
                return True
            if value == "false":
                return False
            return None
        if isinstance(value, int):
            if value == 0:
                return False
            if value == 1:
                return True
        return None
    return None


def casting_parameters(
    args: dict[str, object],
    parameters: dict[str, Parameter]
) -> dict[str, object] | None:
    """
    Cast all arguments into their expected types.

    Args:
        args: The arguments provided for the function call.
        parameters: The parameter definitions with their expected types.

    Returns:
        A dictionary containing the casted arguments,
        or None if a parameter is invalid.
    """
    result: dict[str, object] = {}
    for name, value in args.items():
        if name not in parameters:
            continue
        expected_type = parameters[name].type
        casted_value = cast_parameter(value, expected_type)
        if casted_value is None:
            return None
        result[name] = casted_value
    return result
