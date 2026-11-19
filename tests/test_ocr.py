"""Tests for the recognition pipeline.

The engine is faked throughout. That is deliberate: these tests are about the
shared logic -- ordering, normalisation, correction, confidence, thresholds --
which is where the plate-specific reasoning lives and where a regression would
silently change what the gate does. Engine quality is measured separately, by
the benchmark script, because it needs real images and real weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpr.config.schema import OCRConfig, PlateFormatConfig
from alpr.ocr.base import OCREngine, PlateRecognizer
from alpr.ocr.engines import available_engines, build_engine
from alpr.ocr.postprocess import (
    correct,
    join_lines,
    matches_any_format,
    normalize,
    order_lines,
    template_to_regex,
)
from alpr.ocr.types import TextLine


class FakeEngine(OCREngine):
    """Returns a scripted set of lines, so pipeline behaviour is deterministic."""

    def __init__(self, lines: list[TextLine], *, fail: bool = False) -> None:
        self._lines = lines
        self._fail = fail

    @property
    def name(self) -> str:
        return "fake"

    def read_lines(self, image: np.ndarray) -> list[TextLine]:
        if self._fail:
            raise RuntimeError("engine exploded")
        return list(self._lines)


@pytest.fixture
def formats() -> PlateFormatConfig:
    return PlateFormatConfig()


@pytest.fixture
def ocr_config() -> OCRConfig:
    return OCRConfig(min_char_confidence=0.3, min_string_confidence=0.5)


def make_recognizer(
    lines: list[TextLine],
    ocr_config: OCRConfig,
    formats: PlateFormatConfig,
    *,
    fail: bool = False,
) -> PlateRecognizer:
    return PlateRecognizer(FakeEngine(lines, fail=fail), ocr_config, formats)


CROP = np.zeros((60, 140, 3), dtype=np.uint8)


# --------------------------------------------------------------------------
# normalisation and templates
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" abc-1234 ", "ABC1234"),
        ("51A-961.41", "51A96141"),
        ("nbc 1234", "NBC1234"),
        ("", ""),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_template_to_regex() -> None:
    pattern = template_to_regex("LLLDDDD")
    assert pattern.match("ABC1234")
    assert not pattern.match("AB12345")


def test_matches_any_format(formats: PlateFormatConfig) -> None:
    assert matches_any_format("ABC1234", formats) == "car"
    assert matches_any_format("123ABC", formats) == "motorcycle_new"
    assert matches_any_format("NOTAPLATE", formats) is None


# --------------------------------------------------------------------------
# line ordering -- two-line plates
# --------------------------------------------------------------------------


def test_two_line_plate_is_read_top_row_first() -> None:
    """The bottom row must not come first, whatever order the engine emits."""
    lines = [
        TextLine("2148", 0.9, box=(5, 40, 90, 70)),  # bottom row, emitted first
        TextLine("54S2", 0.9, box=(5, 5, 90, 35)),  # top row
    ]
    assert [line.text for line in order_lines(lines)] == ["54S2", "2148"]


def test_fragments_on_one_row_order_left_to_right() -> None:
    lines = [
        TextLine("1234", 0.9, box=(60, 10, 110, 40)),
        TextLine("ABC", 0.9, box=(5, 10, 55, 40)),
    ]
    assert [line.text for line in order_lines(lines)] == ["ABC", "1234"]


def test_ordering_preserved_without_geometry() -> None:
    lines = [TextLine("ABC", 0.9), TextLine("1234", 0.9)]
    assert [line.text for line in order_lines(lines)] == ["ABC", "1234"]


def test_join_lines_normalises() -> None:
    lines = [TextLine("abc-", 0.9), TextLine(" 1234", 0.9)]
    assert join_lines(lines) == "ABC1234"


# --------------------------------------------------------------------------
# positional correction -- the point of the milestone
# --------------------------------------------------------------------------


def test_clean_read_needs_no_correction(formats: PlateFormatConfig) -> None:
    result = correct("ABC1234", formats)
    assert result is not None
    assert (result.text, result.format_name, result.corrections) == (
        "ABC1234",
        "car",
        0,
    )


def test_letter_in_digit_slot_is_corrected(formats: PlateFormatConfig) -> None:
    """'O' in a digit position is unambiguously a zero."""
    result = correct("ABCO234", formats)
    assert result is not None
    assert result.text == "ABC0234"
    assert result.corrections == 1


def test_digit_in_letter_slot_is_corrected(formats: PlateFormatConfig) -> None:
    """'0' in a letter position is unambiguously an O."""
    result = correct("A0C1234", formats)
    assert result is not None
    assert result.text == "AOC1234"
    assert result.corrections == 1


def test_correction_is_position_dependent(formats: PlateFormatConfig) -> None:
    """The same glyph resolves differently depending on its slot.

    This is the whole argument for templates over regexes: 'O' becomes '0' in a
    digit slot and stays 'O' in a letter slot, within one string.
    """
    result = correct("O0C1234", formats)
    assert result is not None
    assert result.text == "OOC1234"


def test_correction_is_bounded(formats: PlateFormatConfig) -> None:
    """Beyond max_corrections a read is refused rather than forced to fit."""
    # Three letter-O glyphs sitting in digit slots: three substitutions.
    assert correct("ABCOOO4", PlateFormatConfig(max_corrections=1)) is None

    generous = correct("ABCOOO4", PlateFormatConfig(max_corrections=3))
    assert generous is not None
    assert generous.text == "ABC0004"
    assert generous.corrections == 3


def test_letters_in_letter_slots_need_no_correction(
    formats: PlateFormatConfig,
) -> None:
    """'OOO1234' is already valid -- O is a letter and slots 1-3 want letters."""
    result = correct("OOO1234", formats)
    assert result is not None
    assert result.text == "OOO1234"
    assert result.corrections == 0


def test_unmappable_character_fails(formats: PlateFormatConfig) -> None:
    """'X' has no digit equivalent, so it cannot fill a digit slot."""
    assert correct("ABCX234", formats) is None


def test_wrong_length_fails(formats: PlateFormatConfig) -> None:
    assert correct("ABC123", formats) is None


def test_fewest_corrections_wins(formats: PlateFormatConfig) -> None:
    """'123ABC' is a clean motorcycle plate; it must not be mangled into a car."""
    result = correct("123ABC", formats)
    assert result is not None
    assert result.format_name == "motorcycle_new"
    assert result.corrections == 0


def test_correct_is_deterministic(formats: PlateFormatConfig) -> None:
    assert [correct("ABCO234", formats).text for _ in range(20)].count(  # type: ignore[union-attr]
        "ABC0234"
    ) == 20


# --------------------------------------------------------------------------
# pipeline behaviour
# --------------------------------------------------------------------------


def test_end_to_end_read(ocr_config: OCRConfig, formats: PlateFormatConfig) -> None:
    lines = [TextLine("ABC", 0.95, box=(5, 5, 60, 35)), TextLine("1234", 0.9, box=(65, 5, 130, 35))]
    read = make_recognizer(lines, ocr_config, formats).read(CROP)

    assert read.is_valid
    assert read.text == "ABC1234"
    assert read.format_name == "car"
    assert read.engine == "fake"
    assert read.line_count == 2


def test_correction_lowers_confidence(
    ocr_config: OCRConfig, formats: PlateFormatConfig
) -> None:
    """A corrected read must rank below a clean one during voting."""
    clean = make_recognizer(
        [TextLine("ABC1234", 0.9, box=(0, 0, 100, 30))], ocr_config, formats
    ).read(CROP)
    fixed = make_recognizer(
        [TextLine("ABCO234", 0.9, box=(0, 0, 100, 30))], ocr_config, formats
    ).read(CROP)

    assert clean.text == "ABC1234"
    assert fixed.text == "ABC0234"  # O in slot 4 resolves to zero, not one
    assert fixed.confidence < clean.confidence
    assert fixed.corrections == 1


def test_low_confidence_lines_are_dropped(
    ocr_config: OCRConfig, formats: PlateFormatConfig
) -> None:
    lines = [
        TextLine("ABC1234", 0.9, box=(0, 0, 100, 30)),
        TextLine("NOISE", 0.05, box=(0, 40, 100, 70)),
    ]
    read = make_recognizer(lines, ocr_config, formats).read(CROP)
    assert read.text == "ABC1234"


def test_read_below_string_threshold_is_invalid(formats: PlateFormatConfig) -> None:
    strict = OCRConfig(min_char_confidence=0.1, min_string_confidence=0.95)
    read = make_recognizer(
        [TextLine("ABC1234", 0.5, box=(0, 0, 100, 30))], strict, formats
    ).read(CROP)

    assert not read.is_valid
    assert read.text == ""
    assert read.raw_text == "ABC1234"  # preserved for debugging


def test_unrecognisable_text_is_invalid_but_preserved(
    ocr_config: OCRConfig, formats: PlateFormatConfig
) -> None:
    read = make_recognizer(
        [TextLine("HELLO", 0.9, box=(0, 0, 100, 30))], ocr_config, formats
    ).read(CROP)

    assert not read.is_valid
    assert read.raw_text == "HELLO"
    assert read.format_name is None


def test_engine_failure_is_contained(
    ocr_config: OCRConfig, formats: PlateFormatConfig
) -> None:
    """A crashing engine must not take down the frame loop."""
    read = make_recognizer([], ocr_config, formats, fail=True).read(CROP)
    assert not read.is_valid
    assert read.confidence == 0.0


def test_empty_crop_is_handled(
    ocr_config: OCRConfig, formats: PlateFormatConfig
) -> None:
    read = make_recognizer([], ocr_config, formats).read(np.zeros((0, 0, 3), np.uint8))
    assert not read.is_valid


def test_confidence_weighted_by_length(
    ocr_config: OCRConfig, formats: PlateFormatConfig
) -> None:
    """A short high-confidence fragment must not inflate a long uncertain one."""
    lines = [
        TextLine("A", 1.0, box=(0, 0, 10, 30)),
        TextLine("BC1234", 0.6, box=(15, 0, 100, 30)),
    ]
    read = make_recognizer(lines, ocr_config, formats).read(CROP)
    assert read.confidence == pytest.approx((1.0 * 1 + 0.6 * 6) / 7, abs=1e-6)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def test_registry_lists_both_engines() -> None:
    assert set(available_engines()) >= {"paddleocr", "tesseract"}


def test_unknown_engine_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown OCR engine"):
        build_engine(OCRConfig.model_construct(engine="nope", lang="en"))
