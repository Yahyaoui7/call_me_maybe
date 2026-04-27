

def build_function_name_prompt(prompt: str, functions: list) -> str:
    text = "Choose the best function for the user request.\n\n"
    text += f"User request: {prompt}\n\n"
    text += "Available functions:\n"

    for function in functions:
        text += f"- {function.name}: {function.description}\n"

    text += "\nAnswer with only the function name:\n"

    return text


def build_parameters_prompt(prompt: str, function):
    text = "Extract the parameters for this function call.\n"
    text += f"User request: {prompt}\n"
    text += "Selected function:\n"
    text += f"Name: {function.name}\n"
    text += f"Description: {function.description}\n"
    text += "Parameters:\n"

    for name, schema in function.parameters.items():
        text += f"- {name}: {schema.type}\n"

    text += "Return only the parameters as a JSON object.\n"
    text += "Do not include the function name.\n"
    text += "Do not include explanation.\n"

    return text
