from llm_sdk import Small_LLM_Model
import json
from pydantic import ValidationError
from .model import transform_prompts
from .parsing import get_args, parse_files
from .code import ft_encode
from .filter import load_vocab_map, generate_one




def test(prompts_list, full_prompts, model, functions, vocab_map):


    prompt = prompts_list
    full_prompt = full_prompts

    # print("PROMPT:")
    # print(prompt)

    # print("FULL PROMPT:")
    # print(full_prompt)
    i = 0
    for prompt in prompts_list:
        print (f"\n\n this is prompt ::  {prompt}")
        raw_output = generate_one(model, full_prompt[i], functions, vocab_map)
        i += 1
        print("RAW MODEL OUTPUT:")
        print(raw_output)











def main() -> None:
    try:

        args = get_args()
        prompts, functions = parse_files(args.input, args.functions_definition)
        full_prompts, prompts_list = transform_prompts(prompts, functions)
        model = Small_LLM_Model()
        vocap = load_vocab_map(model)
        test(prompts_list, full_prompts, model, functions, vocap)


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
