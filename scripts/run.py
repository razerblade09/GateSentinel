"""run.py — start the real-time ALPR gate system.

Usage
-----
Default config (device=auto, camera=0):

    python scripts/run.py

Point at a video file for offline testing:

    python scripts/run.py --source path/to/clip.mp4

Override any config value from the command line:

    python scripts/run.py --source rtsp://192.168.1.10/stream \\
        --profile local --max-frames 500

Load an allow-list from a plain text file (one plate per line):

    python scripts/run.py --allow-list data/allowed_plates.txt

The script prints one line per gate decision to stdout. Connect it to a gate
controller by piping stdout or reading the database directly.

Output format (stdout, one line per gate decision):

    OPEN  ABC1234  conf=0.91  reads=12  track=3  db=47
    DENY  XYZ9999  conf=0.73  reads=5   track=4  db=48
    DENY           conf=0.00  reads=2   track=5  db=49  (no valid read)

Exit codes
----------
0  Normal exit (file source exhausted, or --max-frames reached).
1  Configuration error or missing file.
2  Runtime error (camera failed to open, model not found, etc.).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alpr.config.loader import load_config
from alpr.pipeline.capture import open_capture
from alpr.pipeline.factory import build_pipeline
from alpr.utils.logging import configure_logging, get_logger


def _load_allow_list(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        sys.exit(f"allow-list file not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    plates = {line.strip().upper() for line in lines if line.strip() and not line.startswith("#")}
    return plates


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Real-time ALPR gate system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--source",
        default=None,
        metavar="SRC",
        help="Camera index, file path, or RTSP URL. Overrides config.",
    )
    p.add_argument(
        "--profile",
        default=None,
        metavar="NAME",
        help="Config profile to load (e.g. 'local', 'kaggle').",
    )
    p.add_argument(
        "--allow-list",
        type=Path,
        default=None,
        metavar="FILE",
        help="Plain text file of authorised plate strings, one per line.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N frames (useful for testing with a video file).",
    )
    p.add_argument(
        "--no-db",
        action="store_true",
        help="Disable database persistence (dry-run mode).",
    )
    p.add_argument(
        "--keep-frames",
        action="store_true",
        help="Attach raw frames to FrameResults (high RAM use, for debugging).",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Display an annotated preview window (requires a display).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # ---- Config ----------------------------------------------------------
    try:
        config = load_config(profile=args.profile)
    except Exception as exc:
        sys.exit(f"Configuration error: {exc}")

    if args.source is not None:
        config.camera.source = args.source  # type: ignore[attr-defined]

    configure_logging(config.logging)
    logger = get_logger(__name__)

    # ---- Allow-list ------------------------------------------------------
    allow_list = _load_allow_list(args.allow_list)
    if allow_list:
        logger.info("allow-list loaded: %d plates", len(allow_list))
    else:
        logger.warning(
            "no allow-list provided; every plate will be DENY (gate never opens)"
        )

    # ---- Pipeline --------------------------------------------------------
    try:
        pipeline = build_pipeline(
            config,
            allow_list=allow_list,
            keep_frames=args.keep_frames,
            with_database=not args.no_db,
        )
    except Exception as exc:
        sys.exit(f"Pipeline build failed: {exc}")

    logger.info(
        "starting capture: source=%r max_fps=%.1f",
        config.camera.source,
        config.runtime.max_fps,
    )

    # ---- Capture loop ----------------------------------------------------
    frames_processed = 0
    try:
        for frame_index, frame in open_capture(
            config.camera, config.runtime, max_frames=args.max_frames
        ):
            result = pipeline.process_frame(frame, frame_index)
            frames_processed += 1

            for decision in result.gate_decisions:
                _print_decision(decision)

            if args.show:
                _show_frame(frame, result)
                import cv2
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        logger.info("interrupted by user")
    finally:
        if args.show:
            import cv2
            cv2.destroyAllWindows()

    logger.info("processed %d frames", frames_processed)


def _print_decision(decision) -> None:  # type: ignore[no-untyped-def]
    status = "OPEN" if decision.gate_opened else "DENY"
    db_part = f"  db={decision.db_event_id}" if decision.db_event_id is not None else ""
    print(
        f"{status:<6}{decision.plate:<12}"
        f"conf={decision.vote.confidence:.2f}  "
        f"reads={decision.vote.n_reads:<4}"
        f"track={decision.vote.track_id}"
        f"{db_part}",
        flush=True,
    )


def _show_frame(frame, result) -> None:  # type: ignore[no-untyped-def]
    """Draw bounding boxes and plate text onto the frame and display it."""
    import cv2

    overlay = frame.copy()
    for vehicle in result.vehicles:
        cv2.rectangle(
            overlay,
            (int(vehicle.x1), int(vehicle.y1)),
            (int(vehicle.x2), int(vehicle.y2)),
            (0, 255, 0), 2,
        )
        if vehicle.track_id is not None:
            cv2.putText(
                overlay, f"T{vehicle.track_id}",
                (int(vehicle.x1), int(vehicle.y1) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
    for plate in result.plates:
        b = plate.plate_box
        color = (0, 128, 255) if plate.ocr_read.is_valid else (0, 0, 180)
        cv2.rectangle(overlay, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
        if plate.ocr_read.text:
            cv2.putText(
                overlay, plate.ocr_read.text,
                (int(b.x1), int(b.y2) + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )
    cv2.imshow("ALPR", overlay)


if __name__ == "__main__":
    main()
