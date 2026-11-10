#!/bin/bash
# Film Developer — main app logic. Lives in the auto-updating engine folder, so fixes ship via GitHub.
# Called by the app bundle with: HOME_DIR, ENGINE, LOG (fd 3 open on the log).
# Written for macOS's built-in bash 3.2.

say() { osascript -e "activate" -e "display dialog \"$1\" buttons {\"OK\"} default button 1 with title \"Film Developer\"" >/dev/null 2>&1; }
note() { osascript -e "display notification \"$1\" with title \"Film Developer\"" >/dev/null 2>&1; }
fail() {
  rm -f "$FD_PROGRESS"
  r=$(osascript -e "activate" -e "try" -e "button returned of (display dialog \"$1\" buttons {\"Show Log\", \"OK\"} default button 2 with title \"Film Developer\")" -e "end try")
  [ "$r" = "Show Log" ] && open -a TextEdit "$LOG"
  exit 1
}
log() { echo "[$(date '+%H:%M:%S')] $*" >&3; }
# progress bar (shown by the app window for anything still running after ~1 second)
export FD_PROGRESS="$HOME_DIR/progress"
pstart() { printf '%s\t%s\t%s\t%s\n' "s$$-$RANDOM" "${3:--1}" "$1" "$2" > "$FD_PROGRESS.tmp" && mv -f "$FD_PROGRESS.tmp" "$FD_PROGRESS"; }
pclear() { rm -f "$FD_PROGRESS"; }

cd "$ENGINE" || exit 1

# ---- only one Film Developer run at a time (a second launch would open a second photo picker)
LOCK="$HOME_DIR/run.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  other="$(cat "$LOCK/pid" 2>/dev/null)"
  if [ -n "$other" ] && kill -0 "$other" 2>/dev/null; then
    osascript -e "display notification \"Film Developer is already open. Look for its window (Mission Control shows all windows).\" with title \"Film Developer\"" >/dev/null 2>&1
    exit 0
  fi
  rm -rf "$LOCK"; mkdir "$LOCK" 2>/dev/null || true      # stale lock from a run that was force-quit
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"; rm -f "$FD_PROGRESS"' EXIT
ARCH="$(uname -m)"                               # x86_64 (Intel) or arm64 (Apple Silicon)
PYVER="3.12"                                     # every component ships ready-made for 3.12 on both
VENV="$HOME_DIR/venv-py$PYVER"
BIN="$HOME_DIR/bin"; UV="$BIN/uv"
export UV_PYTHON_INSTALL_DIR="$HOME_DIR/python"  # keep the app's Python inside its own folder
export UV_CACHE_DIR="$HOME_DIR/cache"
log "launch: engine $(cat VERSION), arch $ARCH, macOS $(sw_vers -productVersion 2>/dev/null), from $([ -n "$FD_APPLET" ] && echo "installed app" || echo "downloaded copy"), installed app v$(sed -n 1p "$HOME_DIR/installed_app" 2>/dev/null)"

# ---- one-time: install a real Mac app (Dock icon, no security prompt) into Applications
APPLET_WANT="$(sed -n 's/^APPLET_VERSION=//p' "$ENGINE/install_app.sh")"
if command -v osacompile >/dev/null 2>&1; then
  inst_ver="$(sed -n 1p "$HOME_DIR/installed_app" 2>/dev/null)"; inst_path="$(sed -n 2p "$HOME_DIR/installed_app" 2>/dev/null)"
  if [ -n "$FD_APPLET" ] && [ -d "$inst_path" ] && [ "$inst_ver" != "$APPLET_WANT" ]; then
    # running from an older installed app: rebuild it, then relaunch the new one once this one quits
    if /bin/bash "$ENGINE/install_app.sh" >/dev/null 2>&3; then
      log "rebuilt installed app (v$APPLET_WANT); relaunching"
      rm -rf "$LOCK"; trap - EXIT
      nohup /bin/bash -c "sleep 4; open \"$inst_path\"" >/dev/null 2>&1 &
      exit 0
    fi
  elif [ -z "$FD_APPLET" ] && [ "$inst_ver" = "$APPLET_WANT" ] && [ -d "$inst_path" ]; then
    # opened from the downloaded copy: hand over to the installed app (it has the progress window)
    log "handing over to installed app at $inst_path"
    rm -rf "$LOCK"; trap - EXIT
    open "$inst_path"
    exit 0
  elif [ -z "$FD_APPLET" ] && { [ "$inst_ver" != "$APPLET_WANT" ] || [ ! -d "$inst_path" ]; }; then
    if APP_PATH="$(/bin/bash "$ENGINE/install_app.sh" 2>&3)" && [ -d "$APP_PATH" ]; then
      log "installed app at $APP_PATH"
      osascript -e "activate" -e "display dialog \"Film Developer is now installed in your Applications folder, with its own Dock icon, and will open from there now. Use that one from now on; you can delete the copy you downloaded.\" buttons {\"OK\"} default button 1 with title \"Film Developer\"" >/dev/null 2>&1
      rm -rf "$LOCK"; trap - EXIT
      open "$APP_PATH"          # hand over to the installed app; this run ends here
      exit 0
    else
      log "app install skipped (osacompile failed)"
    fi
  fi
fi

REQS="requirements.txt"
[ "$ARCH" = "arm64" ] && [ -f requirements-gpu.txt ] && REQS="$REQS requirements-gpu.txt"
REQ_HASH="$(cat $REQS | shasum | cut -d' ' -f1)-$ARCH"

if [ ! -x "$VENV/bin/python" ] || [ "$(cat "$VENV/.req" 2>/dev/null)" != "$REQ_HASH" ]; then
  note "Setting up (first run downloads about 150 MB and takes a few minutes)…"
  pstart "Setting up Film Developer" "Downloading Python and components (first run only, a few minutes)"
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
  pclear
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
pstart "Loading film stocks" "Preparing 21 film stocks"
STOCKS="$("$VENV/bin/python" develop_photo.py --list-tsv 2>&3)" || { pclear; fail "Film Developer couldn't load its film stocks."; }
pclear
NAMES="$(printf '%s\n' "$STOCKS" | cut -f2 | sed 's/"/\\"/g' | awk '{printf "%s\"%s\"", (NR>1?",":""), $0}')"
LAST="$(cat "$HOME_DIR/last_stock" 2>/dev/null || echo "Kodak Portra 400")"
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

# ---- physical look: how the negative is printed
PRINT=$(osascript -e 'activate' -e 'try' -e 'button returned of (display dialog "Print look? (Applies to negative films; slide films are viewed directly.)" buttons {"Photo print", "Cinema print film", "Cinema print + bleach bypass"} default button 1 with title "Film Developer")' -e 'on error' -e 'return ""' -e 'end try')
BLEACH=0
case "$PRINT" in
  "") exit 0 ;;
  "Photo print") MEDIUM=photo_paper ;;
  "Cinema print film") MEDIUM=cinema_2383 ;;
  *) MEDIUM=cinema_2383
     BB=$(osascript -e 'activate' -e 'try' -e 'set r to choose from list {"25% (subtle)", "50%", "75%", "100% (full bypass)"} with title "Film Developer" with prompt "Bleach bypass strength:" default items {"50%"}' -e 'if r is false then return ""' -e 'return item 1 of r' -e 'on error' -e 'return ""' -e 'end try')
     case "$BB" in 25*) BLEACH=0.25 ;; 50*) BLEACH=0.5 ;; 75*) BLEACH=0.75 ;; 100*) BLEACH=1.0 ;; *) exit 0 ;; esac ;;
esac

ARGS=()
while IFS= read -r f; do [ -n "$f" ] && ARGS+=("$f"); done <<< "$FILES"

# ---- develop
note "Developing ${#ARGS[@]} photo(s) on $PICK…"
log "develop: $KEY width $W format $F print $MEDIUM bleach $BLEACH, ${#ARGS[@]} file(s)"
OUTS="$HOME_DIR/outputs.txt"; : > "$OUTS"
set -o pipefail
if "$VENV/bin/python" -u develop_photo.py --stock "$KEY" --width "$W" --format "$F" --print "$MEDIUM" --bleach "$BLEACH" "${ARGS[@]}" 2>&3 \
   | while IFS= read -r line; do
       echo "$line" >&3
       case "$line" in
         PROGRESS*) note "Developing ${line#PROGRESS } …" ;;
         OUTPUT\ *) echo "${line#OUTPUT }" >> "$OUTS" ;;
       esac
     done; then
  pclear
  set +o pipefail
else
  pclear
  fail "Something went wrong developing your photos."
fi

# ---- creative looks: optional finishing window with real-time preview
DEVELOPED=()
while IFS= read -r f; do [ -n "$f" ] && [ -f "$f" ] && DEVELOPED+=("$f"); done < "$OUTS"
[ ${#DEVELOPED[@]} -eq 0 ] && fail "Developing finished but no files were saved."
NEXT=$(osascript -e 'activate' -e 'try' -e 'button returned of (display dialog "Your photos are developed and saved next to the originals. Add a creative look?" buttons {"Show in Finder", "Add a look"} default button 2 with title "Film Developer")' -e 'on error' -e 'return ""' -e 'end try')
if [ "$NEXT" = "Add a look" ]; then
  FIN_LOG="$HOME_DIR/finish.log"; : > "$FIN_LOG"
  "$VENV/bin/python" -u finish_server.py "${DEVELOPED[@]}" > "$FIN_LOG" 2>&3 &
  FIN_PID=$!
  for i in $(seq 1 120); do grep -q FINISH_URL "$FIN_LOG" && break; sleep 0.25; done
  URL="$(sed -n 's/^FINISH_URL //p' "$FIN_LOG" | head -1)"
  if [ -n "$URL" ]; then
    log "finishing window at $URL"
    open "$URL"
    wait "$FIN_PID"                       # until Done is clicked (or the page is closed for 3 minutes)
    cat "$FIN_LOG" >&3
    graded="$(sed -n 's/^SAVED //p' "$FIN_LOG" | tail -1)"
    [ -n "$graded" ] && open -R "$graded" || open -R "${DEVELOPED[0]}"
  else
    kill "$FIN_PID" 2>/dev/null
    fail "The finishing window couldn't start."
  fi
else
  note "Done. Saved next to your originals."
  open -R "${DEVELOPED[0]}"
fi
