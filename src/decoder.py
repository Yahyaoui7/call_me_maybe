import json


def load_vocab_map(model) -> tuple[dict[str, int], dict[int, str]]:
    path_vocab = model.get_path_to_vocab_file()

    with open(path_vocab, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    token_to_id = {k: v for k, v in vocab.items()}
    id_to_token = {v: k for k, v in vocab.items()}

    return token_to_id, id_to_token


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

    prompt_ids = encode_to_ids(model, full_prompt)
    choice_ids = [encode_to_ids(model, choice) for choice in choices]
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



