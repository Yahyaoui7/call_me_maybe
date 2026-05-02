from llm_sdk import Small_LLM_Model
import json
from pathlib import Path
from pydantic import ValidationError
from typing import Any


from .io_utils import get_args, parse_files
from .generator import FunctionCallGenerator
from .prompts import PromptBuilder
from .decoder import Decoder
from .io_utils import FunDef


def get_function(function_name: str, functions: list[FunDef]) -> FunDef:
    for fun in functions:
        if function_name == fun.name:
            return fun
    raise ValueError(f"Unknown function: {function_name}")


def normalize_parameters(
    parameters: dict[str, Any],
    function_def: FunDef,
) -> dict[str, Any]:
    fixed: dict[str, Any] = {}

    for key, value in parameters.items():
        if key not in function_def.parameters:
            raise ValueError(
                f"Unknown parameter '{key}' for function '{function_def.name}'"
            )

        param_type = function_def.parameters[key].type

        if param_type == "number":
            fixed[key] = float(value)
        else:
            fixed[key] = value

    return fixed


def generate_one(
    prompt: str,
    functions: list[Any],
    model: Any,
) -> dict[str, Any]:
    """Generate one valid function-call object for one prompt."""
    prompt_builder = PromptBuilder(prompt)
    decoder = Decoder(model)

    call_generator = FunctionCallGenerator(
        functions=functions,
        prompt_builder=prompt_builder,
        decoder=decoder,
    )

    function_name = call_generator.generate_function_name()

    parameters = call_generator.generate_parameters(
        function_name=function_name,
    )
    fun = get_function(function_name, functions)
    parameters = normalize_parameters(parameters, fun)
    return {
        "prompt": prompt,
        "name": function_name,
        "parameters": parameters,
    }


def transform_prompts(prompts: list[Any]) -> list[Any]:

    prompts_list = []

    for prom in prompts:

        prompts_list.append(prom.prompt)

    return prompts_list


def main() -> None:
    try:
        args = get_args()
        prompts, functions = parse_files(args.input, args.functions_definition)

        prompts_list = transform_prompts(prompts)

        model = Small_LLM_Model()

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            file.write("[\n")

            for index, prompt in enumerate(prompts_list):
                result = generate_one(
                    prompt=prompt,
                    functions=functions,
                    model=model,
                )

                if index > 0:
                    file.write(",\n")

                json.dump(result, file, indent=4)
                file.flush()

            file.write("\n]\n")
    except FileNotFoundError as e:
        print(f"Error: file not found: {e.filename}")
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON at line {e.lineno}, column {e.colno}")
    except ValidationError as e:
        print("Validation error:")
        print(e)
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
