import json
from pydantic import ValidationError

from .parsing import get_args, parse_files


def main() -> None:
    try:
        args = get_args()
        prompts, functions = parse_files(args.input, args.functions_definition)

        print(f"Validated {len(prompts)} prompts.")
        print(f"Validated {len(functions)} function definitions.")
        print(f"Output will be written to: {args.output}")

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
