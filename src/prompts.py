

def build_function_name_prompt(prompt: str, functions: list) -> str:
    text = "Choose the best function for the user request.\n\n"
    text += f"User request: {prompt}\n\n"
    text += "Available functions:\n"

    for function in functions:
        text += f"- {function.name}: {function.description}\n"

    text += "\nAnswer with only the function name:\n"

    return text
