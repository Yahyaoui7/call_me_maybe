



def transform_functions(functions) -> str:
    full_string = "Available functions:\n\n"

    for fun in functions:
        string = f"Function name: {fun.name}\n"
        string += f"Description: {fun.description}\n"
        string += "Parameters:\n"

        for key, value in fun.parameters.items():
            string += f"- {key}: {value.type}\n"

        full_string += string + "\n"

    return full_string


def transform_prompts(prompts, functions) -> list[str]:
    all_prompts = []
    functions_text = transform_functions(functions)
    return_instruction = "Return only a valid JSON object with exactly these keys:\n"
    return_instruction += '- "name"\n- "parameters"\n\n'
    return_instruction += 'Do not include the "prompt" key.\n'
    return_instruction += 'Do not include any explanation, comments, or extra text\n'
    for prom in prompts:
        full_prompt = functions_text
        full_prompt += "User request:\n"
        full_prompt += f"{prom.prompt}\n\n"
        full_prompt += return_instruction

        all_prompts.append(full_prompt)

    return all_prompts
