"""Model export: convert trained .pt weights to ONNX and OpenVINO IR.

The export package is kept separate from the training package because the
two operations have different dependency footprints.  Training needs the
full Ultralytics + Torch stack.  Exporting to OpenVINO additionally needs
``openvino`` (the Python SDK), but it does *not* need a GPU or the CUDA
runtime.  Keeping export separate means the deployment machine only needs
the lighter runtime dependencies.

Typical workflow on the development machine
-------------------------------------------
1. Train on Kaggle → download ``plate_yolo26s.pt``.
2. Run ``python scripts/export.py --weights plate_yolo26s.pt --format openvino``
   to produce ``plate_yolo26s_openvino_model/`` (the IR directory).
3. Copy the IR directory to the gate machine.
4. Set ``vehicle_detector.weights: yolo26n_openvino_model`` and
   ``plate_detector.weights: plate_yolo26s_openvino_model`` in
   ``configs/local.yaml`` on the gate machine.
5. Set ``runtime.device: openvino`` in the same file.
"""

from alpr.export.types import ExportFormat, ExportResult

__all__ = ["ExportFormat", "ExportResult"]
