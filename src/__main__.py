from llm_sdk import Small_LLM_Model
import json
from pydantic import ValidationError
from .model import transform_prompts
from .parsing import get_args, parse_files
from .code import ft_encode
from .filter import load_vocab_map, generate_one
















def main() -> None:
    try:

        args = get_args()
        prompts, functions = parse_files(args.input, args.functions_definition)
        full_prompts = transform_prompts(prompts, functions)
        model = Small_LLM_Model()
        vocap = load_vocab_map(model)
        print(generate_one(model, full_prompts[0], functions, vocap))

    except FileNotFoundError as e:
        print(f"Error: file not found: {e.filename}")
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON at line {e.lineno}, column {e.colno}")
    except ValidationError as e:
        print("Validation error:")
        print(e)
    # except ValueError as e:
    #     print(f"Error,,: {e}")


if __name__ == "__main__":
    main()
