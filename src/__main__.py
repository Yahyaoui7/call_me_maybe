from llm_sdk import Small_LLM_Model
import json
from pathlib import Path
from pydantic import ValidationError
from .model import transform_prompts
from .io_utils import get_args, parse_files
from .generator import generate_one






def main() -> None:
    try:
        args = get_args()
        prompts, functions = parse_files(args.input, args.functions_definition)

        _, prompts_list = transform_prompts(prompts, functions)

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
