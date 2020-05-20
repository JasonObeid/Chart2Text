from typing import Tuple


def mask(s: str, span: Tuple[int, int], mask_token="***mask***") -> Tuple[str, str]:
    masked_sentence = " ".join([s[: span[0]], mask_token, s[span[1] :]])
    return masked_sentence, s[span[0] : span[1]]
