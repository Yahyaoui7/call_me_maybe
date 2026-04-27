

from .decoder import constrained_generate_from_choices

from .prompts import build_function_name_prompt


def generate_function_name(
    prompt: str,
    functions: list,
    model,
) -> str:
    function_names = [function.name for function in functions]

    full_prompt = build_function_name_prompt(prompt, functions)

    return constrained_generate_from_choices(
        model=model,
        full_prompt=full_prompt,
        choices=function_names,
    )


from typing import Any


def generate_one(
    prompt: str,
    functions: list[Any],
    model: Any,
    token_to_id: dict[str, int],
    id_to_token: dict[int, str],
) -> dict[str, Any]:
    """Generate one valid function-call object for one prompt."""

    function_name = generate_function_name(
        prompt=prompt,
        functions=functions,
        model=model,
    )

    function_names = [function.name for function in functions]
    print(function_name)
    if function_name not in function_names:
        raise ValueError(f"Unknown function generated: {function_name}")

    # parameters = generate_parameters(
    #     prompt=prompt,
    #     function_name=function_name,
    #     functions=functions,
    #     model=model,
    #     token_to_id=token_to_id,
    #     id_to_token=id_to_token,
    # )

    return {
        "prompt": prompt,
        "name": function_name,
        # "parameters": parameters,
    }

