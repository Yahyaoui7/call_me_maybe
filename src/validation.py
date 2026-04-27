\


def validate_result(result: dict, function_names: list[str]) -> None:
    if "prompt" not in result:
        raise ValueError("Missing prompt")

    if "name" not in result:
        raise ValueError("Missing name")

    if "parameters" not in result:
        raise ValueError("Missing parameters")

    if result["name"] not in function_names:
        raise ValueError("Invalid function name")
