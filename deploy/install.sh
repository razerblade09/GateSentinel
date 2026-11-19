#!/usr/bin/env bash
# GateSentinel installation script for Raspberry Pi 4/5 or any Debian/Ubuntu
# edge device.
#
# What this script does
# ---------------------
# 1. Installs system dependencies (Python 3.11, libGL, Tesseract).
# 2. Creates a dedicated system user "gatesentinel".
# 3. Clones (or updates) the project to /opt/gatesentinel.
# 4. Creates a virtual environment and installs Python dependencies.
# 5. Installs the systemd service and enables it to start on boot.
#
# Usage
# -----
#   curl -fsSL https://raw.githubusercontent.com/razerblade09/GateSentinel/main/deploy/install.sh | sudo bash
#
# Or, if you have already cloned the repo:
#   sudo bash deploy/install.sh
#
# Prerequisites
# -------------
# * Debian 12 (Bookworm) / Ubuntu 22.04 LTS or later.
# * Internet access for apt and pip (or pre-cached packages).
# * At least 2 GB free disk space (models not included).
#
# After installation
# ------------------
# 1. Copy your exported model files to /opt/gatesentinel/models/:
#      scp models/yolo26n_openvino_model/*  pi@gate:/opt/gatesentinel/models/yolo26n_openvino_model/
#      scp models/plate_yolo26s_openvino_model/*  pi@gate:/opt/gatesentinel/models/plate_yolo26s_openvino_model/
#
# 2. Create /opt/gatesentinel/configs/local.yaml with your camera source:
#      camera:
#        source: "0"           # USB camera index, or an RTSP URL
#
# 3. Start the service:
#      sudo systemctl start gatesentinel
#      sudo journalctl -u gatesentinel -f   # watch logs

set -euo pipefail

INSTALL_DIR=/opt/gatesentinel
REPO_URL=https://github.com/razerblade09/GateSentinel.git
SERVICE_NAME=gatesentinel
VENV_DIR="$INSTALL_DIR/.venv"

log()  { echo "[GateSentinel] $*"; }
die()  { echo "[GateSentinel] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Check we are running as root.
# ---------------------------------------------------------------------------
[[ $EUID -eq 0 ]] || die "This script must be run with sudo."

# ---------------------------------------------------------------------------
# 1. System dependencies.
# ---------------------------------------------------------------------------
log "Installing system dependencies…"
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    git \
    curl

# ---------------------------------------------------------------------------
# 2. Dedicated system user.
# ---------------------------------------------------------------------------
if ! id "$SERVICE_NAME" &>/dev/null; then
    log "Creating system user '$SERVICE_NAME'…"
    useradd --system --no-create-home "$SERVICE_NAME"
else
    log "System user '$SERVICE_NAME' already exists."
fi

# Add to 'video' group so the user can access /dev/video*.
usermod -aG video "$SERVICE_NAME"

# ---------------------------------------------------------------------------
# 3. Clone or update the repository.
# ---------------------------------------------------------------------------
if [[ -d "$INSTALL_DIR/.git" ]]; then
    log "Updating existing installation at $INSTALL_DIR…"
    git -C "$INSTALL_DIR" pull --ff-only
else
    log "Cloning GateSentinel to $INSTALL_DIR…"
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

# ---------------------------------------------------------------------------
# 4. Python virtual environment.
# ---------------------------------------------------------------------------
log "Creating virtual environment at $VENV_DIR…"
python3.11 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip --quiet

log "Installing Python dependencies (this may take a few minutes)…"
"$VENV_DIR/bin/pip" install --quiet \
    numpy>=1.26 \
    "opencv-contrib-python-headless>=4.10" \
    "pydantic>=2.6,<3.0" \
    "pyyaml>=6.0" \
    "sqlalchemy>=2.0,<3.0" \
    "ultralytics>=8.3" \
    "paddleocr>=3.0" \
    "paddlepaddle>=3.0" \
    "openvino>=2024.0"

# ---------------------------------------------------------------------------
# 5. Create required directories.
# ---------------------------------------------------------------------------
mkdir -p "$INSTALL_DIR/models" \
         "$INSTALL_DIR/data" \
         "$INSTALL_DIR/logs"

chown -R "$SERVICE_NAME:$SERVICE_NAME" "$INSTALL_DIR"

# ---------------------------------------------------------------------------
# 6. Systemd service.
# ---------------------------------------------------------------------------
log "Installing systemd service…"
cp "$INSTALL_DIR/deploy/gatesentinel.service" /etc/systemd/system/

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

# ---------------------------------------------------------------------------
# 7. Done — print next steps.
# ---------------------------------------------------------------------------
cat <<'EOF'

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GateSentinel installation complete.

NEXT STEPS:

1. Copy your OpenVINO model files to /opt/gatesentinel/models/
   (run scripts/export.py on your dev machine first, then scp the
   *_openvino_model directories here)

2. Create /opt/gatesentinel/configs/local.yaml:

      camera:
        source: "0"      # USB cam index, or rtsp://... URL
      paths:
        models_dir: /opt/gatesentinel/models
        data_dir:   /opt/gatesentinel/data

3. Start and check the service:

      sudo systemctl start gatesentinel
      sudo journalctl -u gatesentinel -f

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF
