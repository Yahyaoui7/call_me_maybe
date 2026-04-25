
import json


def load_vocab_map(model) -> dict[int, str]:
    path_vocab = model.get_path_to_vocab_file()
    with open(path_vocab, "r", encoding="utf-8") as file:
        vocab = json.load(file)
    return {int(v): k for k, v in vocab.items()}


import json

def _read_field(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _dummy_value(param_schema):
    param_type = _read_field(param_schema, "type", "string")

    if param_type in ("number", "integer"):
        return 0
    if param_type == "boolean":
        return True
    return ""


def _build_candidate_output(fn):
    fn_name = _read_field(fn, "name")
    params_schema = _read_field(fn, "parameters", {})

    params = {}
    for param_name, param_schema in params_schema.items():
        params[param_name] = _dummy_value(param_schema)

    return json.dumps(
        {
            "name": fn_name,
            "parameters": params,
        },
        separators=(",", ":"),
    )


def get_allowed_token_ids(partial_text, functions, chosen_function, vocab_map):
    if chosen_function is not None:
        candidate_outputs = [_build_candidate_output(chosen_function)]
    else:
        candidate_outputs = [_build_candidate_output(fn) for fn in functions]

    allowed_token_ids = []

    for token_id, token_text in vocab_map.items():
        new_text = partial_text + token_text

        if any(candidate.startswith(new_text) for candidate in candidate_outputs):
            allowed_token_ids.append(token_id)

    return allowed_token_ids




def is_finished(partial_text: str) -> bool:
    # starter stop rule
    return len(partial_text) >= 200


def generate_one(model, prompt_text: str, functions, vocab_map: dict[int, str]) -> str:
    generated = ""
    chosen_function = None

    for _ in range(200):
        full_text = prompt_text + generated
        # print("built prompt:", full_text)

        input_ids = model.encode(full_text)[0].tolist()
        logits = model.get_logits_from_input_ids(input_ids)

        allowed_token_ids = get_allowed_token_ids(generated, functions, chosen_function, vocab_map)
        if not allowed_token_ids:
            break

        next_token_id = max(allowed_token_ids, key=lambda token_id: logits[token_id])
        next_token = model.decode([next_token_id])

        # print("next token:", next_token)

        if next_token == "":
            break

        generated += next_token
        print("generated so far:", generated)

        if is_finished(generated):
            break

    return generated
