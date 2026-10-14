#!/usr/bin/env bash
# Set up control of the Scarlett 18i16 4th Gen on Debian 13 (trixie).
#
# The 4th Gen 16i16/18i16/18i20 are controlled through the FCP kernel driver
# (Linux >= 6.14) plus the user-space fcp-server, which creates the ALSA
# controls used by alsa-scarlett-gui, amixer and scarlett.py.
# Debian 13 ships 6.12: the kernel step installs a newer one from backports.
#
# Usage:
#   ./install.sh kernel     # backports kernel (>= 6.14), then REBOOT
#   ./install.sh fcp        # build + install fcp-server / fcp-tool, udev, systemd
#   ./install.sh gui        # build + install alsa-scarlett-gui
#   ./install.sh firmware   # Scarlett 4th Gen firmware files (optional, for updates)
#   ./install.sh all        # everything above (stops after the kernel if a reboot is needed)
#
# Run as your normal user: it calls sudo where needed.

set -euo pipefail

SRC_DIR="${SRC_DIR:-$HOME/.local/src/focusrite}"
FCP_REPO=https://github.com/geoffreybennett/fcp-support.git
GUI_REPO=https://github.com/geoffreybennett/alsa-scarlett-gui.git
FW_REPO=https://github.com/geoffreybennett/scarlett4-firmware.git
CARD="${CARD:-}"   # ALSA card id; empty = the Focusrite card

log() { printf '\n=== %s\n' "$*"; }

kernel_ok() {
    local v
    v=$(uname -r | grep -oE '^[0-9]+\.[0-9]+')
    [ "$(printf '%s\n6.14\n' "$v" | sort -V | head -1)" = "6.14" ]
}

# clone the repo, or update an existing clone
fetch() {
    local repo=$1 dir=$2
    mkdir -p "$SRC_DIR"
    if [ -d "$dir/.git" ]; then
        git -C "$dir" pull --ff-only
    else
        git clone "$repo" "$dir"
    fi
}

step_kernel() {
    if kernel_ok; then
        log "kernel $(uname -r) already >= 6.14"
        return 0
    fi
    log "installing a >= 6.14 kernel from trixie-backports (running: $(uname -r))"
    echo 'deb http://deb.debian.org/debian trixie-backports main' |
        sudo tee /etc/apt/sources.list.d/trixie-backports.list >/dev/null
    sudo apt update
    sudo apt install -y -t trixie-backports linux-image-amd64
    log "kernel installed: REBOOT now, then run: $0 fcp"
    return 1
}

step_fcp() {
    kernel_ok || echo "WARNING: kernel $(uname -r) < 6.14: fcp-server will not find the FCP driver until you run '$0 kernel' and reboot"
    log "building fcp-support"
    sudo apt install -y git build-essential pkg-config alsa-utils \
        libasound2-dev libsystemd-dev libssl-dev zlib1g-dev libjson-c-dev
    fetch "$FCP_REPO" "$SRC_DIR/fcp-support"
    make -C "$SRC_DIR/fcp-support"
    sudo make -C "$SRC_DIR/fcp-support" install

    # `make install` puts the udev rule and tmpfiles under /usr/local/lib, which
    # not every systemd setup reads: copy them to /etc as well
    sudo install -m 644 "$SRC_DIR/fcp-support/udev/99-fcp.rules" /etc/udev/rules.d/99-fcp.rules
    sudo install -m 644 "$SRC_DIR/fcp-support/systemd/fcp-server.tmpfiles.conf" /etc/tmpfiles.d/fcp-server.conf
    sudo systemd-tmpfiles --create /etc/tmpfiles.d/fcp-server.conf

    sudo systemctl daemon-reload
    sudo udevadm control --reload-rules
    # re-trigger the "add" event so the service starts without unplugging the card
    sudo udevadm trigger --action=add --subsystem-match=sound

    if ! id -nG | grep -qw audio; then
        sudo usermod -aG audio "$USER"
        echo "added $USER to group audio: log out and back in"
    fi

    local num
    if [ -n "$CARD" ]; then
        num=$(awk -v c="$CARD" '$2 == "["c {print $1}' /proc/asound/cards | head -1)
    else
        num=$(grep -iE 'Scarlett|Focusrite' /proc/asound/cards | awk 'NR == 1 {print $1}')
    fi
    if [ -n "$num" ]; then
        sleep 1
        systemctl --no-pager status "fcp-server@$num.service" || true
    fi
}

step_gui() {
    log "building alsa-scarlett-gui"
    sudo apt install -y git make gcc libgtk-4-dev libasound2-dev libssl-dev
    fetch "$GUI_REPO" "$SRC_DIR/alsa-scarlett-gui"
    make -C "$SRC_DIR/alsa-scarlett-gui/src" -j"$(nproc)"
    sudo make -C "$SRC_DIR/alsa-scarlett-gui/src" install
}

step_firmware() {
    log "installing Scarlett 4th Gen firmware files to /usr/lib/firmware/scarlett4"
    fetch "$FW_REPO" "$SRC_DIR/scarlett4-firmware"
    sudo mkdir -p /usr/lib/firmware/scarlett4
    sudo cp "$SRC_DIR"/scarlett4-firmware/firmware/*.bin /usr/lib/firmware/scarlett4/
    fcp-tool list-all || true
}

case "${1:-}" in
    kernel) step_kernel || true ;;
    fcp) step_fcp ;;
    gui) step_gui ;;
    firmware) step_firmware ;;
    all)
        step_kernel || exit 0
        step_fcp
        step_gui
        step_firmware
        ;;
    *)
        sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac
