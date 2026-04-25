
import json



def load_vocab_map(model) -> tuple[dict[str, int], dict[int, str]]:
    path_vocab = model.get_path_to_vocab_file()

    with open(path_vocab, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    token_to_id = {k: v for k, v in vocab.items()}
    id_to_token = {v: k for k, v in vocab.items()}

    return token_to_id, id_to_token


def generate_function_name():
    pass


def generate_parameters():
    pass


from typing import Any


def generate_one(
    prompt: str,
    functions: list[Any],
    model: Any,
    token_to_id: dict[str, int],
    id_to_token: dict[int, str],
) -> dict[str, Any]:
    """Generate one valid function-call object for one prompt."""

    function_name = generate_function_name(
        prompt=prompt,
        functions=functions,
        model=model,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
    )

    function_names = [function.name for function in functions]

    if function_name not in function_names:
        raise ValueError(f"Unknown function generated: {function_name}")

    parameters = generate_parameters(
        prompt=prompt,
        function_name=function_name,
        functions=functions,
        model=model,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
    )

    return {
        "prompt": prompt,
        "name": function_name,
        "parameters": parameters,
    }
