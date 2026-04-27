

from .decoder import constrained_generate_from_choices, find_function_by_name, generate_json_text

from .prompts import build_function_name_prompt, build_parameters_prompt
from typing import Any
import json

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

def generate_parameters(
    prompt: str,
    function_name: str,
    functions: list,
    model: Any,
) -> dict[str, Any]:
    """Generate parameters for the selected function."""

    func = find_function_by_name(function_name, functions)
    full_prompt = build_parameters_prompt(prompt, func)

    json_text = generate_json_text(model, full_prompt)
    parameters = json.loads(json_text)

    if not isinstance(parameters, dict):
        raise ValueError("Generated parameters are not a JSON object.")

    return parameters

def generate_one(
    prompt: str,
    functions: list[Any],
    model: Any,
) -> dict[str, Any]:
    """Generate one valid function-call object for one prompt."""

    function_name = generate_function_name(
        prompt=prompt,
        functions=functions,
        model=model,
    )

    function_names = [function.name for function in functions]
    if function_name not in function_names:
        raise ValueError(f"Unknown function generated: {function_name}")

    parameters = generate_parameters(
        prompt=prompt,
        function_name=function_name,
        functions=functions,
        model=model,
    )
    return {
        "prompt": prompt,
        "name": function_name,
        "parameters": parameters,
    }

