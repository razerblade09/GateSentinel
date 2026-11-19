"""Tests for the dataset layer.

The leakage tests matter most. A split bug does not crash -- it silently
inflates every metric the project will ever report, which is exactly the failure
that made the previous model's 0.99 mAP meaningless.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alpr.data.audit import audit_records, group_records, percentiles
from alpr.data.discovery import discover, label_path_for, parse_label_file
from alpr.data.grouping import build_group_index, strip_export_suffixes
from alpr.data.materialize import load_manifest, materialize, save_manifest
from alpr.data.splitting import (
    SplitError,
    split_records,
    verify_no_leakage,
)
from alpr.data.types import BoundingBox, PlateShape

# --------------------------------------------------------------------------
# grouping
# --------------------------------------------------------------------------


def test_strips_roboflow_suffix() -> None:
    assert (
        strip_export_suffixes("quandoi0_jpg.rf.2dcf9d9f273cec58da4f0ce127e72bfd")
        == "quandoi0"
    )


def test_strips_windows_copy_suffix() -> None:
    assert strip_export_suffixes("CarLongPlate248 (1)") == "CarLongPlate248"


def test_offline_augmentation_family_shares_group() -> None:
    """The five offline variants of one photo must collapse to one key."""
    names = [
        "quandoi0_jpg.rf.aaa1.jpg",
        "brightnessquandoi0_jpg.rf.bbb2.jpg",
        "rotatequandoi0_jpg.rf.ccc3.jpg",
        "cropquandoi0_jpg.rf.ddd4.jpg",
        "boderquandoi0_jpg.rf.eee5.jpg",
    ]
    index = build_group_index(names)
    assert len(set(index.values())) == 1


def test_prefix_only_peeled_when_remainder_is_known() -> None:
    """A photo legitimately named 'cropland12' must not be mangled.

    Peeling would produce 'land12', which is not an observed stem, so the
    prefix must be left alone.
    """
    index = build_group_index(["cropland12_jpg.rf.aaa1.jpg", "other_jpg.rf.bbb2.jpg"])
    assert index["cropland12_jpg.rf.aaa1"] == "cropland12"


def test_roboflow_siblings_share_group() -> None:
    index = build_group_index(["photo9_jpg.rf.aaa1.jpg", "photo9_jpg.rf.bbb2.jpg"])
    assert len(set(index.values())) == 1


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("w", "h", "expected"),
    [
        (0.10, 0.08, PlateShape.TWO_LINE),
        (0.20, 0.08, PlateShape.INTERMEDIATE),
        (0.40, 0.08, PlateShape.SINGLE_LINE),
    ],
)
def test_shape_classification(w: float, h: float, expected: PlateShape) -> None:
    assert BoundingBox(0, 0.5, 0.5, w, h).shape() is expected


def test_degenerate_box_detection() -> None:
    assert BoundingBox(0, 0.5, 0.5, 0.0, 0.1).is_degenerate()
    assert BoundingBox(0, 1.5, 0.5, 0.1, 0.1).is_degenerate()
    assert not BoundingBox(0, 0.5, 0.5, 0.1, 0.1).is_degenerate()


def test_percentiles_handle_empty_input() -> None:
    assert percentiles([])["median"] == 0.0


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


def test_label_path_replaces_last_images_component() -> None:
    image = Path("root/plate_images/images/train/a.jpg")
    assert label_path_for(image) == Path("root/plate_images/labels/train/a.txt")


def test_parse_label_file_reports_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("0 0.5 0.5 0.2\n", encoding="utf-8")
    boxes, error = parse_label_file(path)
    assert boxes == []
    assert error is not None and "expected 5+ fields" in error


def _make_dataset(root: Path, spec: dict[str, list[tuple[float, float]]]) -> None:
    """Build a tiny YOLO dataset. ``spec`` maps stem -> list of (w, h)."""
    images = root / "images" / "train"
    labels = root / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for stem, boxes in spec.items():
        (images / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
        lines = [f"0 0.5 0.5 {w} {h}" for w, h in boxes]
        (labels / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def test_discover_pairs_and_reports_orphans(tmp_path: Path) -> None:
    _make_dataset(tmp_path, {"a_jpg.rf.1": [(0.1, 0.08)]})
    (tmp_path / "images" / "train" / "orphan.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "labels" / "train" / "ghost.txt").write_text("0 .5 .5 .1 .1")

    records, issues = discover(tmp_path)

    assert len(records) == 2
    assert len(issues.images_without_labels) == 1
    assert len(issues.labels_without_images) == 1


def test_audit_computes_duplication_factor(tmp_path: Path) -> None:
    _make_dataset(
        tmp_path,
        {
            "p1_jpg.rf.a": [(0.1, 0.08)],
            "p1_jpg.rf.b": [(0.1, 0.08)],
            "p2_jpg.rf.c": [(0.4, 0.08)],
        },
    )
    records, issues = discover(tmp_path)
    audit = audit_records(tmp_path, records, issues)

    assert audit.image_count == 3
    assert audit.group_count == 2
    assert audit.duplication_factor == pytest.approx(1.5)


# --------------------------------------------------------------------------
# splitting -- the important ones
# --------------------------------------------------------------------------


@pytest.fixture
def many_groups(tmp_path: Path) -> Path:
    """200 source photos, each with three augmented copies."""
    spec: dict[str, list[tuple[float, float]]] = {}
    for i in range(200):
        # Vary shape so stratification has something to balance.
        box = (0.1, 0.08) if i % 4 else (0.4, 0.08)
        for variant in ("a", "b", "c"):
            spec[f"photo{i}_jpg.rf.{variant}"] = [box]
    _make_dataset(tmp_path, spec)
    return tmp_path


def test_split_keeps_groups_intact(many_groups: Path) -> None:
    """The core guarantee: no source photo spans two splits."""
    records, _ = discover(many_groups)
    manifest = split_records(records, dataset_root=many_groups)

    verify_no_leakage(manifest)  # raises on violation

    seen: dict[str, str] = {}
    for split, keys in manifest.group_assignments.items():
        for key in keys:
            assert key not in seen, f"{key} leaked between {seen.get(key)} and {split}"
            seen[key] = split


def test_split_is_deterministic(many_groups: Path) -> None:
    records, _ = discover(many_groups)
    first = split_records(records, dataset_root=many_groups, seed=7)
    second = split_records(records, dataset_root=many_groups, seed=7)
    assert first.assignments == second.assignments


def test_different_seed_changes_split(many_groups: Path) -> None:
    records, _ = discover(many_groups)
    first = split_records(records, dataset_root=many_groups, seed=1)
    second = split_records(records, dataset_root=many_groups, seed=2)
    assert first.assignments != second.assignments


def test_split_ratios_are_close_to_target(many_groups: Path) -> None:
    records, _ = discover(many_groups)
    manifest = split_records(records, dataset_root=many_groups)
    counts = manifest.counts()
    total = sum(counts.values())

    assert total == len(records)
    assert counts["train"] / total == pytest.approx(0.70, abs=0.03)
    assert counts["val"] / total == pytest.approx(0.15, abs=0.03)


def test_every_image_appears_exactly_once(many_groups: Path) -> None:
    records, _ = discover(many_groups)
    manifest = split_records(records, dataset_root=many_groups)
    everything = [p for paths in manifest.assignments.values() for p in paths]
    assert len(everything) == len(set(everything)) == len(records)


def test_ratios_must_sum_to_one(many_groups: Path) -> None:
    records, _ = discover(many_groups)
    with pytest.raises(SplitError, match="must sum to 1.0"):
        split_records(
            records,
            dataset_root=many_groups,
            ratios={"train": 0.7, "val": 0.7, "test": 0.15},
        )


def test_too_few_groups_is_refused(tmp_path: Path) -> None:
    _make_dataset(tmp_path, {"only_jpg.rf.a": [(0.1, 0.08)]})
    records, _ = discover(tmp_path)
    with pytest.raises(SplitError, match="too small to divide"):
        split_records(records, dataset_root=tmp_path)


def test_verify_no_leakage_detects_a_planted_violation(many_groups: Path) -> None:
    """Guard against the guard silently passing."""
    records, _ = discover(many_groups)
    manifest = split_records(records, dataset_root=many_groups)

    stolen = manifest.group_assignments["train"][0]
    corrupted = dict(manifest.group_assignments)
    corrupted["val"] = (*corrupted["val"], stolen)
    broken = type(manifest)(
        dataset_root=manifest.dataset_root,
        seed=manifest.seed,
        ratios=manifest.ratios,
        assignments=manifest.assignments,
        group_assignments=corrupted,
        stratum_distribution=manifest.stratum_distribution,
    )

    with pytest.raises(SplitError, match="leakage"):
        verify_no_leakage(broken)


# --------------------------------------------------------------------------
# materialisation
# --------------------------------------------------------------------------


def test_manifest_round_trips(many_groups: Path, tmp_path: Path) -> None:
    records, _ = discover(many_groups)
    manifest = split_records(records, dataset_root=many_groups)

    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)
    restored = load_manifest(path)

    assert restored.assignments == manifest.assignments
    assert restored.seed == manifest.seed


def test_materialize_lists_writes_yaml_and_files(
    many_groups: Path, tmp_path: Path
) -> None:
    records, _ = discover(many_groups)
    manifest = split_records(records, dataset_root=many_groups)

    out = tmp_path / "splits"
    yaml_path = materialize(manifest, out, mode="lists")

    assert yaml_path.is_file()
    for split in ("train", "val", "test"):
        listing = out / f"{split}.txt"
        assert listing.is_file()
        assert len(listing.read_text().strip().splitlines()) == len(
            manifest.assignments[split]
        )


def test_materialize_copy_duplicates_images_and_labels(
    many_groups: Path, tmp_path: Path
) -> None:
    records, _ = discover(many_groups)
    manifest = split_records(records, dataset_root=many_groups)

    out = tmp_path / "copied"
    materialize(manifest, out, mode="copy")

    for split in ("train", "val", "test"):
        images = list((out / "images" / split).glob("*.jpg"))
        labels = list((out / "labels" / split).glob("*.txt"))
        assert len(images) == len(manifest.assignments[split])
        assert len(labels) == len(images)


def test_val_never_points_at_train(many_groups: Path, tmp_path: Path) -> None:
    """Regression guard against the defect that voided the previous model."""
    import yaml as pyyaml

    records, _ = discover(many_groups)
    manifest = split_records(records, dataset_root=many_groups)
    out = tmp_path / "splits"
    yaml_path = materialize(manifest, out, mode="lists")

    document = pyyaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert document["val"] != document["train"]

    train_images = set((out / "train.txt").read_text().split())
    val_images = set((out / "val.txt").read_text().split())
    assert not (train_images & val_images)
