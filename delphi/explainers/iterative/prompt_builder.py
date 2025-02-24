from delphi.explainers.iterative.prompts import SYSTEM


def build_prompt(
    explanation: str,
    normal_examples: str,
    false_positives: str,
    false_negatives: str,
) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM}]

    user_start = f"Current explanation: {explanation}\n\n"
    user_start += f"Normal examples:\n{normal_examples}\n\n"
    user_start += f"False positives:\n{false_positives}\n\n"
    user_start += f"False negatives:\n{false_negatives}\n"

    messages.append(
        {
            "role": "user",
            "content": user_start,
        }
    )

    return messages
