
import json


def load_vocab_map(model) -> dict[int, str]:
    path_vocab = model.get_path_to_vocab_file()
    with open(path_vocab, "r", encoding="utf-8") as file:
        vocab = json.load(file)
    return {int(v): k for k, v in vocab.items()}


def get_allowed_token_ids(
    partial_text: str,
    functions,
    chosen_function,
    vocab_map: dict[int, str],
) -> list[int]:
    # starter version: allow all tokens
    # later you will filter only valid JSON/schema tokens
    return list(vocab_map.keys())


def is_finished(partial_text: str) -> bool:
    # starter stop rule
    return len(partial_text) >= 200


def generate_one(
    model,
    prompt_text: str,
    functions,
    vocab_map: dict[int, str],
) -> str:
    generated = ""
    chosen_function = None

    for _ in range(200):
        full_text = prompt_text + generated
        input_ids = model.encode(full_text)[0].tolist()
        logits = model.get_logits_from_input_ids(input_ids)

        allowed_token_ids = get_allowed_token_ids(
            generated,
            functions,
            chosen_function,
            vocab_map,
        )

        if not allowed_token_ids:
            break

        next_token_id = max(
            allowed_token_ids,
            key=lambda token_id: logits[token_id]
        )

        next_token = model.decode([next_token_id])

        if next_token == "":
            break

        generated += next_token

        if is_finished(generated):
            break

    return generated
