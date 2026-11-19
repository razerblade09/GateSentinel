"""OCR evaluation metrics: Character Error Rate, exact match, and format rate.

Three metrics, each answering a different question.

**Character Error Rate (CER)**
    ``edit_distance(prediction, ground_truth) / len(ground_truth)``.
    The normalisation makes it comparable across plates of different lengths.
    CER = 0 means a perfect read. CER = 0.14 means roughly one character in
    seven is wrong. The edit distance is computed with standard dynamic
    programming (Wagner-Fischer), not a string library, so there are no
    hidden case-folding or Unicode normalisation surprises.

**Exact-match accuracy**
    Fraction of plates where the prediction matches the ground truth exactly,
    after the same normalisation applied by the production pipeline. This is
    the ``did it open the right gate'' metric. CER is better for
    understanding *how* the system is failing; exact match is better for
    reporting *whether* it works.

**Format-match rate**
    Fraction of predictions that satisfy any known plate format. This is a
    proxy for ``did the pipeline produce a plausible string at all'', which
    is useful when ground-truth labels are unavailable. A high format-match
    rate with low exact-match accuracy means the pipeline is reading plates
    confidently but reading the wrong ones.
"""

from __future__ import annotations

from dataclasses import dataclass

from alpr.config.schema import PlateFormatConfig
from alpr.ocr.postprocess import matches_any_format, normalize


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OCRSample:
    """One plate crop's prediction and ground truth.

    Attributes:
        image_id: Arbitrary identifier for error messages and aggregation.
        prediction: The pipeline's output string. Use ``""`` when the
            pipeline produced no read (failed confidence threshold, format
            mismatch, etc.). Do **not** pass ``None`` -- distinguishing
            "no read" from "empty plate" is the caller's job.
        ground_truth: The correct plate string. Must be non-empty. Apply the
            same normalisation as the production pipeline before passing it
            in, or call :func:`normalize` here.
        engine: OCR engine name, for per-engine breakdowns.
    """

    image_id: str
    prediction: str
    ground_truth: str
    engine: str = ""


@dataclass
class OCRSummary:
    """Aggregate OCR performance over a dataset.

    Attributes:
        n_samples: Number of plates evaluated.
        n_reads: Number of non-empty predictions (pipeline produced a string).
        mean_cer: Mean Character Error Rate over all samples. Empty predictions
            contribute a CER of 1.0 (every character wrong).
        exact_match_rate: Fraction of samples where prediction == ground truth.
        format_match_rate: Fraction of non-empty predictions that satisfy a
            known plate format. ``None`` when no :class:`PlateFormatConfig`
            was provided.
        read_rate: Fraction of samples where the pipeline produced any string.
        per_engine: Breakdown by engine name, each value being a sub-summary.
            Only populated when samples carry engine names.
    """

    n_samples: int
    n_reads: int
    mean_cer: float
    exact_match_rate: float
    format_match_rate: float | None
    read_rate: float
    per_engine: dict[str, "OCRSummary"]

    def __str__(self) -> str:
        fmt = (
            f", fmt={self.format_match_rate:.3f}"
            if self.format_match_rate is not None
            else ""
        )
        return (
            f"OCRSummary(n={self.n_samples}, "
            f"CER={self.mean_cer:.3f}, "
            f"exact={self.exact_match_rate:.3f}, "
            f"read_rate={self.read_rate:.3f}"
            f"{fmt})"
        )


# ---------------------------------------------------------------------------
# Edit distance (Wagner-Fischer)
# ---------------------------------------------------------------------------


def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein edit distance between two strings.

    Uses the two-row rolling array so memory is O(min(|a|, |b|)) rather than
    O(|a| * |b|). Plate strings are short (<12 chars), so this is fast
    enough without any further optimisation.

    >>> _edit_distance("ABC1234", "ABC0234")
    1
    >>> _edit_distance("", "ABC1234")
    7
    >>> _edit_distance("ABC1234", "ABC1234")
    0
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Keep the shorter string as columns to minimise memory.
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr

    return prev[len(b)]


# ---------------------------------------------------------------------------
# Per-sample metric
# ---------------------------------------------------------------------------


def character_error_rate(prediction: str, ground_truth: str) -> float:
    """CER for one (prediction, ground_truth) pair.

    Both strings should already be normalised (uppercase, no separators).
    Call :func:`alpr.ocr.postprocess.normalize` on raw pipeline output before
    passing it in.

    Args:
        prediction: Pipeline output. ``""`` for a failed read.
        ground_truth: Correct plate string. Must be non-empty.

    Returns:
        Edit distance / len(ground_truth), clamped to [0, 1]. Returns 1.0 for
        an empty prediction (every character wrong). Returns 0.0 for a perfect
        match.

    Raises:
        ValueError: If ``ground_truth`` is empty -- CER is undefined when
            there is no reference.

    >>> character_error_rate("ABC1234", "ABC1234")
    0.0
    >>> character_error_rate("", "ABC1234")
    1.0
    >>> round(character_error_rate("ABC0234", "ABC1234"), 4)
    0.1429
    """
    if not ground_truth:
        raise ValueError("ground_truth must be non-empty; CER is undefined otherwise")

    if not prediction:
        return 1.0

    dist = _edit_distance(prediction, ground_truth)
    return min(1.0, dist / len(ground_truth))


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------


def _summarise(
    samples: list[OCRSample],
    format_config: PlateFormatConfig | None,
) -> OCRSummary:
    """Build a summary over ``samples``.  Called recursively for per-engine."""
    n = len(samples)
    if n == 0:
        return OCRSummary(
            n_samples=0,
            n_reads=0,
            mean_cer=0.0,
            exact_match_rate=0.0,
            format_match_rate=None,
            read_rate=0.0,
            per_engine={},
        )

    cer_sum = 0.0
    exact = 0
    fmt_ok = 0
    n_reads = 0

    for s in samples:
        pred = normalize(s.prediction)
        gt = normalize(s.ground_truth)
        cer_sum += character_error_rate(pred, gt)
        if pred == gt:
            exact += 1
        if pred:
            n_reads += 1
            if format_config is not None and matches_any_format(pred, format_config):
                fmt_ok += 1

    fmt_rate: float | None = (
        (fmt_ok / n_reads) if (format_config is not None and n_reads > 0) else None
    )

    return OCRSummary(
        n_samples=n,
        n_reads=n_reads,
        mean_cer=cer_sum / n,
        exact_match_rate=exact / n,
        format_match_rate=fmt_rate,
        read_rate=n_reads / n,
        per_engine={},
    )


def ocr_metrics(
    samples: list[OCRSample],
    format_config: PlateFormatConfig | None = None,
) -> OCRSummary:
    """Aggregate OCR metrics over a dataset.

    Args:
        samples: One :class:`OCRSample` per plate crop. Ground truths must
            already be normalised to uppercase alphanumeric.
        format_config: Optional plate format rules. When provided,
            :attr:`OCRSummary.format_match_rate` is computed.

    Returns:
        :class:`OCRSummary` with dataset-level metrics and optional per-engine
        breakdown (populated whenever any sample carries an engine name).
    """
    summary = _summarise(samples, format_config)

    # Per-engine breakdown -- only when at least one engine label is present.
    engines = {s.engine for s in samples if s.engine}
    if engines:
        summary.per_engine = {
            engine: _summarise(
                [s for s in samples if s.engine == engine], format_config
            )
            for engine in sorted(engines)
        }

    return summary
