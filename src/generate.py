
def choose_best_allowed_token(logits, allowed_ids: set[int]) -> int:
    """Choose the allowed token with the highest score."""

    best_id = None
    best_score = None

    for token_id in allowed_ids:
        score = logits[token_id]

        if best_score is None or score > best_score:
            best_score = score
            best_id = token_id

    if best_id is None:
        raise ValueError("Could not choose token.")

    return best_id

def get_allowed_next_ids(
    generated_ids: list[int],
    choice_ids: list[list[int]],
) -> set[int]:
    """Return token IDs that can continue a valid choice."""

    allowed_ids: set[int] = set()

    for ids in choice_ids:
        if ids[:len(generated_ids)] == generated_ids:
            if len(ids) > len(generated_ids):
                allowed_ids.add(ids[len(generated_ids)])

    return allowed_ids



def constrained_generate_from_choices(
    model,
    full_prompt: str,
    choices: list[str],
) -> str:
    """Generate only one value from choices."""

    prompt_ids = model.encode(full_prompt)
    choice_ids = [model.encode(choice) for choice in choices]

    generated_ids: list[int] = []

    while True:
        # If generated tokens equal one full choice, stop.
        for choice, ids in zip(choices, choice_ids):
            if generated_ids == ids:
                return choice

        allowed_ids = get_allowed_next_ids(
            generated_ids=generated_ids,
            choice_ids=choice_ids,
        )

        if not allowed_ids:
            raise ValueError("No valid token allowed.")

        input_ids = prompt_ids + generated_ids

        logits = model.get_logits_from_input_ids(input_ids)

        next_id = choose_best_allowed_token(
            logits=logits,
            allowed_ids=allowed_ids,
        )

        generated_ids.append(next_id)


def build_function_name_prompt(prompt: str, functions: list) -> str:
    text = "Choose the best function for the user request.\n\n"
    text += f"User request: {prompt}\n\n"
    text += "Available functions:\n"

    for function in functions:
        text += f"- {function.name}: {function.description}\n"

    text += "\nAnswer with only the function name:\n"

    return text



def generate_function_name(
    prompt: str,
    functions: list,
    model,
) -> str:
    """Choose one valid function name using the LLM."""

    function_names = [function.name for function in functions]

    full_prompt = build_function_name_prompt(prompt, functions)

    return constrained_generate_from_choices(
        model=model,
        full_prompt=full_prompt,
        choices=function_names,
    )



def build_parameters_prompt():
    pass
