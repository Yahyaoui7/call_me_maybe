
from typing import Any


class Decoder:

    def __init__(self, model: Any) -> None:
        self.model = model


    def encode_to_ids(self, text: str) -> list[int]:
        ids = self.model.encode(text)

        if hasattr(ids, "tolist"):
            ids = ids.tolist()

        if ids and isinstance(ids[0], list):
            ids = ids[0]

        return ids



    def get_allowed_next_ids(self,
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


    def choose_best_allowed_token(self, logits, allowed_ids: set[int]) -> int:
        """Choose the allowed token with the highest score."""

        best_id = None
        best_score = None
        scores = self.get_last_logits(logits)
        for token_id in allowed_ids:

            score = scores[token_id]

            if best_score is None or score > best_score:
                best_score = score
                best_id = token_id

        if best_id is None:
            raise ValueError("Could not choose token.")

        return best_id



    def constrained_generate_from_choices(self,
        full_prompt: str,
        choices: list[str],
    ) -> str:
        """Generate only one value from choices."""
        prompt_ids = self.encode_to_ids( full_prompt)
        choice_ids = [self.encode_to_ids( choice) for choice in choices]
        generated_ids: list[int] = []

        while True:
            # If generated tokens equal one full choice, stop.
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

    # = = =  prameters decoding  = = =


    def get_last_logits(self, logits: Any) -> list[float]:
        """Convert self.model logits to a simple list of scores."""

        if hasattr(logits, "tolist"):
            logits = logits.tolist()

        while isinstance(logits, list) and logits and isinstance(logits[0], list):
            logits = logits[-1]

        return logits


    def choose_best_token(self, logits: Any) -> int:
        """Choose the token with the highest score."""

        scores = self.get_last_logits(logits)

        best_id = 0
        best_score = scores[0]

        for token_id, score in enumerate(scores):
            if score > best_score:
                best_score = score
                best_id = token_id

        return best_id


    def generate_json_text(self, full_prompt: str, max_tokens: int = 30) -> str:
        """Ask the LLM to generate JSON text."""

        prompt_ids = self.encode_to_ids(full_prompt)
        generated_ids: list[int] = []

        for _ in range(max_tokens):
            input_ids = prompt_ids + generated_ids

            logits = self.model.get_logits_from_input_ids(input_ids)
            next_id = self.choose_best_token(logits)

            generated_ids.append(next_id)

            generated_text = self.model.decode(generated_ids)
            start = generated_text.find("{")
            end = generated_text.find("}")
            if start != -1 and end != -1 and end > start:
                return generated_text[start:end + 1]


        raise ValueError("Could not generate parameters JSON.")





