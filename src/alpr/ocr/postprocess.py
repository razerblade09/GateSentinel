"""Turning raw OCR output into a validated plate string.

Three steps, each of which the previous system either skipped or got wrong.

**Line ordering.** An engine emits text regions in no particular order. A
two-line plate must be reassembled top row first, then left to right within each
row. Getting this wrong silently transposes the halves of every motorcycle plate.

**Normalisation.** Strip separators and whitespace, uppercase. Philippine plates
are written ``ABC-1234``; Vietnamese ones use ``.`` -- neither separator carries
information worth keeping.

**Positional correction.** This is the part that was missing entirely. A
Philippine car plate is ``LLLDDDD``: three letters, four digits. If slot 4 must
be a digit and OCR returned ``O``, that is a ``0`` -- deterministically, with no
guessing. The old pipeline rejected such reads instead of repairing them, which
threw away every plate that was one confusable glyph away from correct.

The correction is bounded on purpose. Given unlimited substitutions any string
of the right length can be forced into a valid format, which at a gate means
confidently opening the barrier for noise. Failing loudly is the better outcome.
"""

from __future__ import annotations

import re

from alpr.config.schema import PlateFormatConfig
from alpr.ocr.types import CorrectionResult, TextLine

_NON_ALNUM = re.compile(r"[^A-Z0-9]")


def normalize(text: str) -> str:
    """Uppercase and strip everything that is not a letter or digit.

    >>> normalize(" abc-1234 ")
    'ABC1234'
    """
    return _NON_ALNUM.sub("", text.upper())


def order_lines(lines: list[TextLine], row_tolerance: float = 0.5) -> list[TextLine]:
    """Sort text lines into human reading order.

    Lines whose vertical centres fall within ``row_tolerance`` of the median
    line height are treated as the same row. Without this, two fragments of a
    single row can be ordered by ``y`` alone and end up interleaved.

    Args:
        lines: Engine output, in arbitrary order.
        row_tolerance: Row grouping threshold, as a fraction of line height.

    Returns:
        The same lines, ordered top-to-bottom then left-to-right.
    """
    if len(lines) <= 1:
        return list(lines)
    if any(line.box is None for line in lines):
        # No geometry available: preserve the engine's own ordering.
        return list(lines)

    heights = sorted(line.height for line in lines)
    median_height = heights[len(heights) // 2] or 1.0
    threshold = median_height * row_tolerance

    by_vertical = sorted(lines, key=lambda line: line.y_center)

    rows: list[list[TextLine]] = [[by_vertical[0]]]
    for line in by_vertical[1:]:
        if abs(line.y_center - rows[-1][-1].y_center) <= threshold:
            rows[-1].append(line)
        else:
            rows.append([line])

    ordered: list[TextLine] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda line: line.x_start))
    return ordered


def join_lines(lines: list[TextLine]) -> str:
    """Concatenate ordered lines into one normalised candidate string."""
    return normalize("".join(line.text for line in lines))


def _apply_template(
    text: str,
    template: str,
    letter_to_digit: dict[str, str],
    digit_to_letter: dict[str, str],
) -> tuple[str, int] | None:
    """Coerce ``text`` to ``template``, returning the result and edit count.

    Returns ``None`` when a character cannot be coerced at all -- for example a
    letter in a digit slot with no known confusion mapping. That is a genuine
    mismatch rather than a correctable one.
    """
    if len(text) != len(template):
        return None

    out: list[str] = []
    corrections = 0

    for char, slot in zip(text, template, strict=True):
        if slot == "L":
            if char.isalpha():
                out.append(char)
            elif char in digit_to_letter:
                out.append(digit_to_letter[char])
                corrections += 1
            else:
                return None
        else:  # slot == "D"
            if char.isdigit():
                out.append(char)
            elif char in letter_to_digit:
                out.append(letter_to_digit[char])
                corrections += 1
            else:
                return None

    return "".join(out), corrections


def correct(text: str, config: PlateFormatConfig) -> CorrectionResult | None:
    """Find the plate format this text matches with the fewest corrections.

    Args:
        text: Normalised candidate string.
        config: Region format rules.

    Returns:
        The best :class:`CorrectionResult`, or ``None`` if no format matches
        within ``config.max_corrections``.
    """
    if not text:
        return None

    best: CorrectionResult | None = None

    for name, template in sorted(config.formats.items()):
        applied = _apply_template(
            text, template, config.letter_to_digit, config.digit_to_letter
        )
        if applied is None:
            continue
        corrected, corrections = applied
        if corrections > config.max_corrections:
            continue
        # Ties broken by format name so the result is deterministic rather than
        # dependent on dict ordering.
        if best is None or corrections < best.corrections:
            best = CorrectionResult(
                text=corrected, format_name=name, corrections=corrections
            )

    return best


def template_to_regex(template: str) -> re.Pattern[str]:
    """Derive the validation regex for a slot template.

    Deriving rather than storing a second source of truth: validation and
    correction can then never disagree about what a valid plate looks like.

    >>> bool(template_to_regex("LLLDDDD").match("ABC1234"))
    True
    """
    body = "".join("[A-Z]" if slot == "L" else "[0-9]" for slot in template)
    return re.compile(f"^{body}$")


def matches_any_format(text: str, config: PlateFormatConfig) -> str | None:
    """Return the name of the format ``text`` already satisfies, if any."""
    for name, template in sorted(config.formats.items()):
        if template_to_regex(template).match(text):
            return name
    return None
