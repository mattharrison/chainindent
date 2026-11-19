"""Apply replacements to offsets in an original source string."""

from collections.abc import Iterable


type Edit = tuple[int, int, str]


def apply_edits(text: str, right_to_left: Iterable[Edit]) -> str:
    """Apply rightmost edits first, skipping spans that overlap accepted edits."""
    parts = []
    limit = len(text)
    for start, end, replacement in right_to_left:
        if end <= limit:
            parts.extend((text[end:limit], replacement))
            limit = start
    parts.append(text[:limit])
    return "".join(reversed(parts))
