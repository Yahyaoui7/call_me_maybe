import json
from typing import Any


# def load_vocab_map(model) -> tuple[dict[str, int], dict[int, str]]:
#     path_vocab = model.get_path_to_vocab_file()

#     with open(path_vocab, "r", encoding="utf-8") as file:
#         vocab = json.load(file)

#     token_to_id = {k: v for k, v in vocab.items()}
#     id_to_token = {v: k for k, v in vocab.items()}

#     return token_to_id, id_to_token


def encode_to_ids(model, text: str) -> list[int]:
    ids = model.encode(text)

    if hasattr(ids, "tolist"):
        ids = ids.tolist()

    if ids and isinstance(ids[0], list):
        ids = ids[0]

    return ids



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



def constrained_generate_from_choices(
    model,
    full_prompt: str,
    choices: list[str],
) -> str:
    """Generate only one value from choices."""
    print(full_prompt)
    prompt_ids = encode_to_ids(model, full_prompt)
    choice_ids = [encode_to_ids(model, choice) for choice in choices]
    generated_ids: list[int] = []

    while True:
        # If generated tokens equal one full choice, stop.
        for choice, ids in zip(choices, choice_ids):
            if generated_ids == ids:
                return choice

        allowed_ids = get_allowed_next_ids(
            generated_ids,
            choice_ids,
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

# = = =  prameters decoding  = = =

def find_function_by_name(function_name: str, functions: list):

    for function in functions :
        if function.name == function_name:
            return function
    raise ValueError(f"Function not found {function_name}")




def get_last_logits(logits: Any) -> list[float]:
    """Convert model logits to a simple list of scores."""

    if hasattr(logits, "tolist"):
        logits = logits.tolist()

    while isinstance(logits, list) and logits and isinstance(logits[0], list):
        logits = logits[-1]

    return logits


def choose_best_token(logits: Any) -> int:
    """Choose the token with the highest score."""

    scores = get_last_logits(logits)

    best_id = 0
    best_score = scores[0]

    for token_id, score in enumerate(scores):
        if score > best_score:
            best_score = score
            best_id = token_id

    return best_id


def generate_json_text(model: Any, full_prompt: str, max_tokens: int = 80) -> str:
    """Ask the LLM to generate JSON text."""

    prompt_ids = encode_to_ids(model, full_prompt)
    generated_ids: list[int] = []

    for _ in range(max_tokens):
        input_ids = prompt_ids + generated_ids

        logits = model.get_logits_from_input_ids(input_ids)
        next_id = choose_best_token(logits)

        generated_ids.append(next_id)

        generated_text = model.decode(generated_ids)

        start = generated_text.find("{")
        end = generated_text.find("}")

        if start != -1 and end != -1 and end > start:
            return generated_text[start:end + 1]

    raise ValueError("Could not generate parameters JSON.")
