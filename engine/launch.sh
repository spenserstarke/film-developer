#!/bin/bash
# Film Developer — main app logic. Lives in the auto-updating engine folder, so fixes ship via GitHub.
# Called by the app bundle with: HOME_DIR, ENGINE, LOG (fd 3 open on the log).
# Written for macOS's built-in bash 3.2.

say() { osascript -e "activate" -e "display dialog \"$1\" buttons {\"OK\"} default button 1 with title \"Film Developer\"" >/dev/null 2>&1; }
note() { osascript -e "display notification \"$1\" with title \"Film Developer\"" >/dev/null 2>&1; }
fail() {
  r=$(osascript -e "activate" -e "try" -e "button returned of (display dialog \"$1\" buttons {\"Show Log\", \"OK\"} default button 2 with title \"Film Developer\")" -e "end try")
  [ "$r" = "Show Log" ] && open -a TextEdit "$LOG"
  exit 1
}
log() { echo "[$(date '+%H:%M:%S')] $*" >&3; }

cd "$ENGINE" || exit 1
ARCH="$(uname -m)"                               # x86_64 (Intel) or arm64 (Apple Silicon)
PYVER="3.12"                                     # every component ships ready-made for 3.12 on both
VENV="$HOME_DIR/venv-py$PYVER"
BIN="$HOME_DIR/bin"; UV="$BIN/uv"
export UV_PYTHON_INSTALL_DIR="$HOME_DIR/python"  # keep the app's Python inside its own folder
export UV_CACHE_DIR="$HOME_DIR/cache"
log "launch: engine $(cat VERSION), arch $ARCH, macOS $(sw_vers -productVersion 2>/dev/null)"

REQS="requirements.txt"
[ "$ARCH" = "arm64" ] && [ -f requirements-gpu.txt ] && REQS="$REQS requirements-gpu.txt"
REQ_HASH="$(cat $REQS | shasum | cut -d' ' -f1)-$ARCH"

if [ ! -x "$VENV/bin/python" ] || [ "$(cat "$VENV/.req" 2>/dev/null)" != "$REQ_HASH" ]; then
  note "Setting up (first run downloads about 150 MB and takes a few minutes)…"
  # 1) uv: a small tool that installs a private copy of Python and the components
  if [ ! -x "$UV" ]; then
    case "$ARCH" in arm64) UVARCH=aarch64 ;; *) UVARCH=x86_64 ;; esac
    TMP="$(mktemp -d)"
    log "downloading uv ($UVARCH)"
    if curl -fsSL --max-time 120 "https://github.com/astral-sh/uv/releases/latest/download/uv-$UVARCH-apple-darwin.tar.gz" -o "$TMP/uv.tgz" >&3 2>&1 \
       && tar -xzf "$TMP/uv.tgz" -C "$TMP" >&3 2>&1; then
      mkdir -p "$BIN" && cp "$TMP"/uv-*/uv "$UV" && chmod +x "$UV"
    fi
    rm -rf "$TMP"
  fi
  ok=0
  if [ -x "$UV" ]; then
    log "creating Python $PYVER environment with uv"
    rm -rf "$VENV"
    if "$UV" venv --python "$PYVER" "$VENV" >&3 2>&1 \
       && "$UV" pip install --only-binary :all: --python "$VENV/bin/python" -r requirements.txt >&3 2>&1; then
      ok=1
      if [ "$ARCH" = "arm64" ] && [ -f requirements-gpu.txt ]; then
        "$UV" pip install --only-binary :all: --python "$VENV/bin/python" -r requirements-gpu.txt >&3 2>&1 \
          || log "GPU components unavailable; CPU engine will be used"
      fi
    fi
  fi
  # 2) fallback: a Python already on this Mac (3.10–3.13; 3.14 lacks ready-made RAW support on Intel)
  if [ "$ok" = 0 ]; then
    log "uv route failed; trying a Python already installed"
    PY=""
    for c in python3.12 python3.13 python3.11 python3.10 \
             /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
             /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
             /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
      command -v "$c" >/dev/null 2>&1 && { PY="$(command -v "$c")"; break; }
    done
    if [ -n "$PY" ]; then
      rm -rf "$VENV"
      "$PY" -m venv "$VENV" >&3 2>&1 && "$VENV/bin/python" -m pip install -q --upgrade pip >&3 2>&1 \
        && "$VENV/bin/python" -m pip install --only-binary=:all: -r requirements.txt >&3 2>&1 && ok=1
    fi
  fi
  [ "$ok" = 1 ] || fail "Setup couldn't download its components. Check your internet connection and open Film Developer again. If it keeps failing, click Show Log and send the log to Claude."
  echo "$REQ_HASH" > "$VENV/.req"
  log "setup complete"
fi

# (--only-binary: never compile from source. Newer releases sometimes drop Intel-Mac builds, e.g.
#  rawpy 0.28; with this flag the installer picks the newest version that has a ready-made build.)

# ---- choose photos
FILES=$(osascript 2>&3 <<'EOF'
activate
try
  set picks to choose file with prompt "Choose photos to develop (RAW works best; JPEG, PNG and TIFF also work)" with multiple selections allowed
  set out to ""
  repeat with f in picks
    set out to out & POSIX path of f & linefeed
  end repeat
  return out
on error
  return ""
end try
EOF
)
[ -z "$FILES" ] && exit 0

# ---- choose stock
STOCKS="$("$VENV/bin/python" develop_photo.py --list-tsv 2>&3)" || fail "Film Developer couldn't load its film stocks."
NAMES="$(printf '%s\n' "$STOCKS" | cut -f2 | sed 's/"/\\"/g' | awk '{printf "%s\"%s\"", (NR>1?",":""), $0}')"
LAST="$(cat "$HOME_DIR/last_stock" 2>/dev/null || echo "Kodak Portra 400")"
note "Loading film stocks…"
log "stocks loaded: $(printf '%s\n' "$STOCKS" | grep -c .)"
PICK=$(osascript -e "activate" -e "try" -e "set r to choose from list {$NAMES} with title \"Film Developer\" with prompt \"Film stock:\" default items {\"$LAST\"}" -e "if r is false then return \"\"" -e "return item 1 of r" -e "on error m" -e "return \"ERROR: \" & m" -e "end try" 2>&3)
log "stock picked: $PICK"
case "$PICK" in "") exit 0 ;; ERROR:*) fail "The film stock list couldn't open." ;; esac
echo "$PICK" > "$HOME_DIR/last_stock"
KEY="$(printf '%s\n' "$STOCKS" | awk -F'\t' -v n="$PICK" '$2==n {print $1}')"

SIZE=$(osascript -e 'activate' -e 'try' -e 'button returned of (display dialog "Output size?" buttons {"Quick (1600 px)", "Standard (2400 px)", "Large (4000 px)"} default button 2 with title "Film Developer")' -e 'on error' -e 'return ""' -e 'end try')
case "$SIZE" in Quick*) W=1600 ;; Large*) W=4000 ;; "") exit 0 ;; *) W=2400 ;; esac
FMT=$(osascript -e 'activate' -e 'try' -e 'button returned of (display dialog "Film format?" buttons {"35mm", "120 medium format"} default button 1 with title "Film Developer")' -e 'on error' -e 'return ""' -e 'end try')
case "$FMT" in 120*) F=120 ;; "") exit 0 ;; *) F=35 ;; esac

ARGS=()
while IFS= read -r f; do [ -n "$f" ] && ARGS+=("$f"); done <<< "$FILES"

# ---- develop
note "Developing ${#ARGS[@]} photo(s) on $PICK…"
log "develop: $KEY width $W format $F, ${#ARGS[@]} file(s)"
set -o pipefail
if "$VENV/bin/python" -u develop_photo.py --stock "$KEY" --width "$W" --format "$F" "${ARGS[@]}" 2>&3 \
   | while IFS= read -r line; do
       echo "$line" >&3
       case "$line" in PROGRESS*) note "Developing ${line#PROGRESS } …" ;; esac
     done; then
  note "Done. Saved next to your originals."
  first="${ARGS[0]%.*}_$KEY"
  for ext in tif jpg png; do [ -f "$first.$ext" ] && { open -R "$first.$ext"; break; }; done
else
  fail "Something went wrong developing your photos."
fi
