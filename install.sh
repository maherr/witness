#!/usr/bin/env bash
# install.sh
#
# Installs Witness on a Linux x86_64 machine, any distribution:
#   - transcribe.cpp (pinned commit) with the Parakeet Ultra Q8_0 model
#   - whisper.cpp (pinned commit, CPU build) for Silero speech detection and
#     the English/French label; none of its words reach a transcript
#   - speakrs-cli (pinned commit) with a stock ONNX Runtime for speaker labels
#   - the `witness` command and its runtime config
#
# Nothing here needs a particular GPU vendor or a patched library.
# CPU is the floor. If the Vulkan build files and a hardware Vulkan device are
# present, Parakeet Ultra runs on the GPU. Speaker labels run on CPU unless you
# bring an ONNX Runtime with the MIGraphX provider (WITNESS_ORT_LIB, AMD only).
#
# Overridable env vars:
#   PREFIX            Runtimes and models          (default: ~/.local/share/witness)
#   BIN_DIR           Where `witness` is installed (default: ~/.local/bin)
#   WITNESS_GPU       auto | vulkan | cpu          (default: auto)
#   WITNESS_ORT_LIB   Use this libonnxruntime.so.1.24.2 instead of downloading one
#   JOBS              Parallel build jobs          (default: nproc)
#   SKIP_CHECKS       Skip the prerequisite check  (default: unset)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PREFIX="${PREFIX:-$HOME/.local/share/witness}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
WITNESS_GPU="${WITNESS_GPU:-auto}"
JOBS="${JOBS:-$(nproc)}"
RUNTIME_CONFIG="${WITNESS_RUNTIME_CONFIG:-$HOME/.config/witness/runtime.json}"

TRANSCRIBE_REPO="https://github.com/handy-computer/transcribe.cpp.git"
TRANSCRIBE_REF="5d66f60257a75e9bdbddc72dc0887b32e9d4841a"
WHISPER_CPP_REPO="https://github.com/ggerganov/whisper.cpp.git"
WHISPER_CPP_REF="dc9611662265870df22a7230b7586176a99c1955"
SPEAKRS_REPO="${SPEAKRS_REPO:-https://github.com/maherr/speakrs.git}"
SPEAKRS_REF="${SPEAKRS_REF:-05d07ef28940f281efab20256ed853114a7ac25f}"

ULTRA_MODEL_REV="a1539be1b849ea7463b88343e5277b373d0c4ea9"
ULTRA_MODEL_URL="https://huggingface.co/handy-computer/parakeet-ultra-gguf/resolve/$ULTRA_MODEL_REV/parakeet-ultra-Q8_0.gguf"
LANGUAGE_MODEL_REV="5359861c739e955e79d9a303bcbc70fb988958b1"
LANGUAGE_MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/$LANGUAGE_MODEL_REV/ggml-base.bin"
VAD_MODEL_REV="9ffd54a1e1ee413ddf265af9913beaf518d1639b"
VAD_MODEL_URL="https://huggingface.co/ggml-org/whisper-vad/resolve/$VAD_MODEL_REV/ggml-silero-v6.2.0.bin"
ORT_VERSION="1.24.2"
MIN_RUST="1.85"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/onnxruntime-linux-x64-$ORT_VERSION.tgz"

TRANSCRIBE_SRC="$PREFIX/src/transcribe.cpp"
TRANSCRIBE_PREFIX="$PREFIX/transcribe.cpp/${TRANSCRIBE_REF:0:8}"
WHISPER_SRC="$PREFIX/src/whisper.cpp"
SPEAKRS_SRC="$PREFIX/src/speakrs"
MODEL_DIR="$PREFIX/models"
ORT_DIR="$PREFIX/onnxruntime-$ORT_VERSION"

ULTRA_CLI="$TRANSCRIBE_PREFIX/bin/transcribe-cli"
ULTRA_MODEL="$MODEL_DIR/parakeet-ultra-Q8_0.gguf"
WHISPER_CLI="$WHISPER_SRC/build/bin/whisper-cli"
VAD_CLI="$WHISPER_SRC/build/bin/whisper-vad-speech-segments"
LANGUAGE_MODEL="$MODEL_DIR/ggml-base.bin"
VAD_MODEL="$MODEL_DIR/ggml-silero-v6.2.0.bin"
SPEAKRS_CLI="$BIN_DIR/speakrs-cli"

USE_VULKAN=0
ULTRA_BACKEND="cpu"
DIARIZER="cpu"
ORT_LIB=""

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m==>\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m==>\033[0m %s\n' "$1" >&2; exit "${2:-1}"; }

distro_family() {
  local id="" like=""
  if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    id="$(. /etc/os-release && printf '%s' "${ID:-}")"
    # shellcheck disable=SC1091
    like="$(. /etc/os-release && printf '%s' "${ID_LIKE:-}")"
  fi
  case " $id $like " in
    *" arch "*) echo arch ;;
    *" fedora "*|*" rhel "*) echo fedora ;;
    *" debian "*|*" ubuntu "*) echo debian ;;
    *) echo unknown ;;
  esac
}

print_hints() {
  warn ""
  case "$(distro_family)" in
    fedora)
      warn "Fedora:"
      warn "  sudo dnf install git cmake make gcc gcc-c++ curl tar python3 ffmpeg pkgconf-pkg-config openssl-devel cargo"
      warn "  GPU (optional): sudo dnf install vulkan-headers vulkan-loader-devel glslc spirv-headers-devel"
      ;;
    debian)
      warn "Debian / Ubuntu:"
      warn "  sudo apt install git cmake make build-essential curl tar python3 ffmpeg pkg-config libssl-dev"
      warn "  GPU (optional): sudo apt install libvulkan-dev glslc spirv-headers"
      warn "  Rust $MIN_RUST or newer (the packaged one is often too old):"
      warn "    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
      ;;
    arch)
      warn "Arch (and Arch-based, such as Omarchy):"
      warn "  sudo pacman -S --needed git cmake make gcc curl tar python ffmpeg pkgconf openssl rust"
      warn "  GPU (optional): sudo pacman -S --needed vulkan-headers vulkan-icd-loader shaderc spirv-headers"
      ;;
    *)
      warn "Install with your package manager: git, cmake, make, a C and C++ compiler,"
      warn "curl, tar, python3, ffmpeg, pkg-config, and the OpenSSL development files."
      warn "GPU (optional): the Vulkan headers and loader development files, and glslc."
      warn "Rust $MIN_RUST or newer:"
      warn "    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
      ;;
  esac
}

rust_new_enough() {
  local version
  version="$(rustc --version 2>/dev/null | awk '{print $2}')"
  [ -n "$version" ] || return 1
  [ "$(printf '%s\n%s\n' "$MIN_RUST" "$version" | sort -V | head -1)" = "$MIN_RUST" ]
}

vulkan_buildable() {
  # The loader alone is not enough: ffmpeg pulls it in on some distributions
  # without the headers.
  command -v glslc >/dev/null 2>&1 \
    && command -v pkg-config >/dev/null 2>&1 \
    && pkg-config --exists vulkan \
    && [ -f "$(pkg-config --variable=includedir vulkan)/vulkan/vulkan.h" ]
}

check_prereqs() {
  [ "$(uname -s)" = "Linux" ] || die "This installer supports Linux only" 1
  if [ "$(uname -m)" != "x86_64" ] && [ -z "${WITNESS_ORT_LIB:-}" ]; then
    die "Only x86_64 is supported and tested. On another architecture, provide your own ONNX Runtime $ORT_VERSION through WITNESS_ORT_LIB and expect to debug." 1
  fi
  case "$WITNESS_GPU" in
    auto|vulkan|cpu) ;;
    *) die "WITNESS_GPU must be auto, vulkan, or cpu" 1 ;;
  esac

  if [ -n "${SKIP_CHECKS:-}" ]; then
    warn "Skipping prerequisite checks (SKIP_CHECKS set)"
  else
    log "Checking prerequisites"
    local missing=() tool
    for tool in git cmake make cc c++ curl tar python3 ffmpeg ffprobe pkg-config sha256sum; do
      command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
    done
    if ! command -v cargo >/dev/null 2>&1 || ! command -v rustc >/dev/null 2>&1; then
      missing+=("rust-toolchain")
    elif ! rust_new_enough; then
      missing+=("rust-$MIN_RUST-or-newer (found $(rustc --version | awk '{print $2}'))")
    fi
    if command -v pkg-config >/dev/null 2>&1 && ! pkg-config --exists openssl; then
      missing+=("openssl-development")
    fi
    if [ "$WITNESS_GPU" = "vulkan" ] && ! vulkan_buildable; then
      missing+=("vulkan-development (headers, loader, glslc)")
    fi
    if [ ${#missing[@]} -gt 0 ]; then
      warn "Missing: ${missing[*]}"
      print_hints
      die "Install the missing pieces and re-run" 1
    fi
    log "Prerequisites OK"
  fi

  if [ "$WITNESS_GPU" != "cpu" ] && vulkan_buildable; then
    USE_VULKAN=1
  elif [ "$WITNESS_GPU" = "auto" ]; then
    warn "Vulkan build files not found: building for CPU only."
    warn "Install the optional GPU packages and re-run to use a GPU for transcription."
  fi
}

checkout_pinned_ref() {
  local dir="$1" repo="$2" ref="$3" label="$4"
  if [ ! -d "$dir/.git" ]; then
    log "Cloning $label"
    mkdir -p "$(dirname "$dir")"
    git clone --quiet "$repo" "$dir" || die "$label clone failed" 2
  fi
  if ! git -C "$dir" diff --quiet || ! git -C "$dir" diff --cached --quiet; then
    die "$label source tree is dirty at $dir; clean it before a reproducible rebuild" 2
  fi
  if ! git -C "$dir" cat-file -e "$ref^{commit}" 2>/dev/null; then
    git -C "$dir" fetch --quiet origin "$ref" \
      || die "Failed to fetch pinned $label commit $ref" 2
  fi
  git -C "$dir" checkout --detach --quiet "$ref" \
    || die "Failed to check out pinned $label commit $ref" 2
}

verify_download() {
  local path="$1" expected_size="$2" expected_sha="$3"
  [ -f "$path" ] || return 1
  [ "$(stat -c%s "$path")" -eq "$expected_size" ] || return 1
  printf '%s  %s\n' "$expected_sha" "$path" | sha256sum --check --status
}

download_verified() {
  local url="$1" path="$2" expected_size="$3" expected_sha="$4"
  if verify_download "$path" "$expected_size" "$expected_sha"; then
    log "Verified existing $(basename "$path")"
    return
  fi
  mkdir -p "$(dirname "$path")"
  if [ -f "$path" ] && [ "$(stat -c%s "$path")" -ge "$expected_size" ]; then
    rm -f "$path"
  fi
  log "Downloading $(basename "$path")"
  curl -L --fail --retry 3 --progress-bar --continue-at - \
    -o "$path" "$url" || { rm -f "$path"; die "Download failed: $url" 2; }
  if verify_download "$path" "$expected_size" "$expected_sha"; then
    return
  fi
  warn "Downloaded file failed size/SHA verification; retrying once from scratch"
  rm -f "$path"
  curl -L --fail --retry 3 --progress-bar \
    -o "$path" "$url" || { rm -f "$path"; die "Download retry failed: $url" 2; }
  verify_download "$path" "$expected_size" "$expected_sha" \
    || { rm -f "$path"; die "Downloaded file failed verification: $url" 2; }
}

configure_transcribe_cpp() {
  local build_dir="$1" vulkan_flag="$2"
  rm -rf "$build_dir"
  mkdir -p "$build_dir"
  # A static build: one self-contained transcribe-cli, nothing to keep on a
  # library path. The CPU backend is compiled for this machine's processor, so
  # rebuild on another machine instead of copying the result.
  cmake -S "$TRANSCRIBE_SRC" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRANSCRIBE_BUILD_SHARED=OFF \
    -DTRANSCRIBE_BUILD_EXAMPLES=ON \
    -DTRANSCRIBE_BUILD_TESTS=OFF \
    -DTRANSCRIBE_INSTALL=OFF \
    -DTRANSCRIBE_USE_SYSTEM_BLAS=OFF \
    -DTRANSCRIBE_VULKAN="$vulkan_flag" >"$build_dir/witness-configure.log" 2>&1
}

transcribe_cpp_built() {
  [ -x "$ULTRA_CLI" ] \
    && [ "$(cat "$TRANSCRIBE_PREFIX/.witness-build" 2>/dev/null)" = "$TRANSCRIBE_REF vulkan=$1" ]
}

build_transcribe_cpp() {
  local vulkan_flag="OFF" build_dir="$TRANSCRIBE_SRC/build-witness"
  [ "$USE_VULKAN" = "1" ] && vulkan_flag="ON"
  if transcribe_cpp_built "$vulkan_flag"; then
    log "transcribe.cpp already built (Vulkan $vulkan_flag)"
    return
  fi
  checkout_pinned_ref "$TRANSCRIBE_SRC" "$TRANSCRIBE_REPO" "$TRANSCRIBE_REF" "transcribe.cpp"
  if ! configure_transcribe_cpp "$build_dir" "$vulkan_flag"; then
    if [ "$vulkan_flag" = "OFF" ] || [ "$WITNESS_GPU" = "vulkan" ]; then
      tail -30 "$build_dir/witness-configure.log" >&2
      die "transcribe.cpp configure failed" 3
    fi
    # Some part of the Vulkan build files is missing in a way the earlier
    # check did not see. CPU is the floor, so say why and carry on.
    warn "The Vulkan build could not be configured:"
    grep -m 3 -A 1 'CMake Error' "$build_dir/witness-configure.log" >&2 || true
    warn "Building for CPU only. Install the optional GPU packages and re-run to use a GPU."
    USE_VULKAN=0
    vulkan_flag="OFF"
    if transcribe_cpp_built OFF; then
      log "transcribe.cpp already built (Vulkan OFF)"
      return
    fi
    configure_transcribe_cpp "$build_dir" OFF \
      || { tail -30 "$build_dir/witness-configure.log" >&2; die "transcribe.cpp configure failed" 3; }
  fi
  log "Building transcribe.cpp (Vulkan $vulkan_flag, $JOBS jobs)"
  cmake --build "$build_dir" -j "$JOBS" --target transcribe-cli >"$build_dir/witness-build.log" 2>&1 \
    || { tail -30 "$build_dir/witness-build.log" >&2; die "transcribe.cpp build failed" 3; }
  [ -x "$build_dir/bin/transcribe-cli" ] \
    || die "transcribe.cpp built, but $build_dir/bin/transcribe-cli is missing" 3
  install -Dm755 "$build_dir/bin/transcribe-cli" "$ULTRA_CLI"
  printf '%s vulkan=%s\n' "$TRANSCRIBE_REF" "$vulkan_flag" > "$TRANSCRIBE_PREFIX/.witness-build"
  log "transcribe-cli → $ULTRA_CLI"
}

choose_ultra_backend() {
  if [ "$USE_VULKAN" != "1" ]; then
    return
  fi
  # ggml's Vulkan backend skips software renderers, so a listed Vulkan device
  # is a real GPU. Capture first: under pipefail, `grep -q` closing the pipe
  # early would read as "no device".
  local devices
  devices="$("$ULTRA_CLI" --list-devices 2>&1 || true)"
  if grep -q 'kind=vulkan' <<<"$devices"; then
    ULTRA_BACKEND="vulkan"
    log "Hardware Vulkan device found: Parakeet Ultra will run on the GPU"
  elif [ "$WITNESS_GPU" = "vulkan" ]; then
    die "WITNESS_GPU=vulkan, but transcribe-cli lists no hardware Vulkan device" 3
  else
    warn "No hardware Vulkan device found: Parakeet Ultra will run on CPU."
  fi
}

build_whisper_cpp() {
  if [ -x "$WHISPER_CLI" ] && [ -x "$VAD_CLI" ] \
      && [ "$(git -C "$WHISPER_SRC" rev-parse HEAD 2>/dev/null)" = "$WHISPER_CPP_REF" ]; then
    log "whisper.cpp helpers already built"
    return
  fi
  checkout_pinned_ref "$WHISPER_SRC" "$WHISPER_CPP_REPO" "$WHISPER_CPP_REF" "whisper.cpp"
  log "Building whisper.cpp helpers for CPU ($JOBS jobs)"
  cmake -S "$WHISPER_SRC" -B "$WHISPER_SRC/build" -DCMAKE_BUILD_TYPE=Release \
    >"$WHISPER_SRC/build.configure.log" 2>&1 \
    || { tail -30 "$WHISPER_SRC/build.configure.log" >&2; die "whisper.cpp configure failed" 3; }
  cmake --build "$WHISPER_SRC/build" -j "$JOBS" --target whisper-cli whisper-vad-speech-segments \
    >"$WHISPER_SRC/build.build.log" 2>&1 \
    || { tail -30 "$WHISPER_SRC/build.build.log" >&2; die "whisper.cpp build failed" 3; }
}

build_speakrs() {
  if [ -x "$SPEAKRS_CLI" ] \
      && [ "$(git -C "$SPEAKRS_SRC" rev-parse HEAD 2>/dev/null)" = "$SPEAKRS_REF" ] \
      && [ "$SPEAKRS_SRC/target/release/speakrs-cli" -ot "$SPEAKRS_CLI" ]; then
    log "speakrs-cli already built"
    return
  fi
  checkout_pinned_ref "$SPEAKRS_SRC" "$SPEAKRS_REPO" "$SPEAKRS_REF" "speakrs"
  log "Building speakrs-cli ($JOBS jobs)"
  cargo build --release --locked --bin speakrs-cli --jobs "$JOBS" \
    --manifest-path "$SPEAKRS_SRC/Cargo.toml" >"$SPEAKRS_SRC/build.log" 2>&1 \
    || { tail -30 "$SPEAKRS_SRC/build.log" >&2; die "speakrs-cli build failed" 3; }
  install -Dm755 "$SPEAKRS_SRC/target/release/speakrs-cli" "$SPEAKRS_CLI"
  log "speakrs-cli → $SPEAKRS_CLI"
}

install_onnxruntime() {
  if [ -n "${WITNESS_ORT_LIB:-}" ]; then
    [ -f "$WITNESS_ORT_LIB" ] || die "WITNESS_ORT_LIB does not exist: $WITNESS_ORT_LIB" 1
    ORT_LIB="$WITNESS_ORT_LIB"
    # A provider library beside it means speaker labels can try the AMD GPU.
    if [ -f "$(dirname "$ORT_LIB")/libonnxruntime_providers_migraphx.so" ]; then
      DIARIZER="auto"
    fi
    log "Using your ONNX Runtime: $ORT_LIB (speaker labels: $DIARIZER)"
    return
  fi
  local archive="$PREFIX/downloads/onnxruntime-linux-x64-$ORT_VERSION.tgz"
  ORT_LIB="$ORT_DIR/lib/libonnxruntime.so.$ORT_VERSION"
  if [ ! -f "$ORT_LIB" ]; then
    download_verified "$ORT_URL" "$archive" 8123282 \
      "43725474ba5663642e17684717946693850e2005efbd724ac72da278fead25e6"
    mkdir -p "$ORT_DIR"
    tar -xzf "$archive" -C "$ORT_DIR" --strip-components=1 \
      || die "Could not unpack ONNX Runtime" 2
  fi
  [ -f "$ORT_LIB" ] || die "ONNX Runtime library missing after unpack: $ORT_LIB" 2
  log "ONNX Runtime $ORT_VERSION (CPU) → $ORT_DIR"
}

download_models() {
  download_verified "$ULTRA_MODEL_URL" "$ULTRA_MODEL" 740363168 \
    "007a59761e9258779f189df396b50d12b83bb5b79e66df4b955c230d2f2a0a59"
  download_verified "$LANGUAGE_MODEL_URL" "$LANGUAGE_MODEL" 147951465 \
    "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe"
  download_verified "$VAD_MODEL_URL" "$VAD_MODEL" 885098 \
    "2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987"
}

install_witness() {
  install -Dm755 "$SCRIPT_DIR/witness" "$BIN_DIR/witness"
  mkdir -p "$(dirname "$RUNTIME_CONFIG")"
  python3 - "$RUNTIME_CONFIG" \
    ultra_cli "$ULTRA_CLI" \
    ultra_model "$ULTRA_MODEL" \
    ultra_backend "$ULTRA_BACKEND" \
    whisper_cli "$WHISPER_CLI" \
    vad_cli "$VAD_CLI" \
    language_model "$LANGUAGE_MODEL" \
    vad_model "$VAD_MODEL" \
    speakrs_cli "$SPEAKRS_CLI" \
    ort_lib "$ORT_LIB" \
    diarizer "$DIARIZER" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
pairs = sys.argv[2:]
config = {}
try:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        config = loaded
except (OSError, ValueError):
    pass
config.update(zip(pairs[0::2], pairs[1::2]))
path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY
  chmod 600 "$RUNTIME_CONFIG"
  log "witness → $BIN_DIR/witness"
  log "Runtime config → $RUNTIME_CONFIG"
}

summary() {
  echo
  log "Witness is installed."
  log "  Transcription:  Parakeet Ultra on $ULTRA_BACKEND"
  log "  Speaker labels: speakrs on $([ "$DIARIZER" = "auto" ] && echo 'AMD GPU with CPU fallback' || echo CPU)"
  case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) warn "$BIN_DIR is not on your PATH; add it or call $BIN_DIR/witness directly." ;;
  esac
  echo
  log "Try it on the sample that ships with whisper.cpp:"
  log "  $BIN_DIR/witness $WHISPER_SRC/samples/jfk.wav --output /tmp/jfk.md && cat /tmp/jfk.md"
}

main() {
  log "Installing Witness into $PREFIX"
  check_prereqs
  build_transcribe_cpp
  choose_ultra_backend
  build_whisper_cpp
  build_speakrs
  install_onnxruntime
  download_models
  install_witness
  summary
}

main "$@"
