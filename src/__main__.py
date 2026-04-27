from llm_sdk import Small_LLM_Model
import json
from pathlib import Path
from pydantic import ValidationError
from .model import transform_prompts
from .io_utils import get_args, parse_files
from .decoder import load_vocab_map
from .generator import generate_one






def main() -> None:
    try:

        args = get_args()
        prompts, functions = parse_files(args.input, args.functions_definition)
        full_prompts, prompts_list = transform_prompts(prompts, functions)
        model = Small_LLM_Model()
        token_to_id, id_to_token = load_vocab_map(model)
        results = []

        for prompt in prompts_list:
            result = generate_one(
                prompt=prompt,
                functions=functions,
                model=model,
                token_to_id=token_to_id,
                id_to_token=id_to_token,
            )

            results.append(result)

        # output_path = Path(args.output)
        # output_path.parent.mkdir(parents=True, exist_ok=True)
        # print(dir(output_path))
        # with open(output_path, "w", encoding="utf-8") as file:
        #     json.dump(results, file, indent=4)


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
