from .prompts import SYSTEM


def build_prompt(
    explanation: str,
    normal_examples: str,
    wrong_examples: str,
) -> list[dict]:
    messages = SYSTEM

    user_start = f"Current explanation: {explanation}\n\nNormal examples:\n{normal_examples}\n\nWrong examples:\n{wrong_examples}\n"

    messages.append(
        {
            "role": "user",
            "content": user_start,
        }
    )

    return messages

