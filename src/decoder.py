from typing import Any


class Decoder:

    def __init__(self, model: Any) -> None:
        self.model = model

    def encode_to_ids(self, text: str) -> list[int]:
        ids_obj = self.model.encode(text)

        if hasattr(ids_obj, "tolist"):
            ids_obj = ids_obj.tolist()

        if (
            isinstance(ids_obj, list)
            and ids_obj
            and isinstance(ids_obj[0], list)
        ):
            ids_obj = ids_obj[0]

        if not isinstance(ids_obj, list):
            raise TypeError("Model encode must return a list of token IDs.")

        return [int(token_id) for token_id in ids_obj]

    def get_allowed_next_ids(
        self,
        generated_ids: list[int],
        choice_ids: list[list[int]],
    ) -> set[int]:
        allowed_ids: set[int] = set()

        for ids in choice_ids:
            if ids[: len(generated_ids)] == generated_ids:
                if len(ids) > len(generated_ids):
                    allowed_ids.add(ids[len(generated_ids)])

        return allowed_ids

    def choose_best_allowed_token(
        self,
        logits: Any,
        allowed_ids: set[int],
    ) -> int:
        best_id: int | None = None
        best_score: float | None = None
        scores = self.get_last_logits(logits)

        for token_id in allowed_ids:
            score = scores[token_id]

            if best_score is None or score > best_score:
                best_score = score
                best_id = token_id

        if best_id is None:
            raise ValueError("Could not choose token.")

        return best_id

    def constrained_generate_from_choices(
        self,
        full_prompt: str,
        choices: list[str],
    ) -> str:
        prompt_ids = self.encode_to_ids(full_prompt)
        choice_ids = [self.encode_to_ids(choice) for choice in choices]
        generated_ids: list[int] = []

        while True:
            for choice, ids in zip(choices, choice_ids):
                if generated_ids == ids:
                    return choice

            allowed_ids = self.get_allowed_next_ids(
                generated_ids,
                choice_ids,
            )

            if not allowed_ids:
                raise ValueError("No valid token allowed.")

            input_ids = prompt_ids + generated_ids
            logits = self.model.get_logits_from_input_ids(input_ids)

            next_id = self.choose_best_allowed_token(
                logits=logits,
                allowed_ids=allowed_ids,
            )

            generated_ids.append(next_id)

    def get_last_logits(self, logits: Any) -> list[float]:
        logits_obj = logits

        if hasattr(logits_obj, "tolist"):
            logits_obj = logits_obj.tolist()

        while (
            isinstance(logits_obj, list)
            and logits_obj
            and isinstance(logits_obj[0], list)
        ):
            logits_obj = logits_obj[-1]

        if not isinstance(logits_obj, list):
            raise TypeError("Logits must be a list of scores.")

        return [float(score) for score in logits_obj]

    def choose_best_token(self, logits: Any) -> int:
        scores = self.get_last_logits(logits)

        best_id = 0
        best_score = scores[0]

        for token_id, score in enumerate(scores):
            if score > best_score:
                best_score = score
                best_id = token_id

        return best_id

    def generate_json_text(
        self,
        full_prompt: str,
        max_tokens: int = 200,
    ) -> Any:
        prompt_ids = self.encode_to_ids(full_prompt)
        generated_ids: list[int] = []

        started = False
        brace_count = 0
        inside_string = False
        escape_next = False

        for _ in range(max_tokens):
            input_ids = prompt_ids + generated_ids
            logits = self.model.get_logits_from_input_ids(input_ids)
            next_id = self.choose_best_token(logits)
            generated_ids.append(next_id)

            generated_text = self.model.decode(generated_ids)

            started = False
            brace_count = 0
            inside_string = False
            escape_next = False

            for i, char in enumerate(generated_text):
                if escape_next:
                    escape_next = False
                    continue

                if char == "\\" and inside_string:
                    escape_next = True
                    continue

                if char == '"':
                    inside_string = not inside_string
                    continue

                if not inside_string:
                    if char == "{":
                        started = True
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if started and brace_count == 0:
                            return generated_text[: i + 1]

        raise ValueError("Could not generate complete JSON.")
