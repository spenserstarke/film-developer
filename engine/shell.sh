#!/bin/bash
# Film Developer — updater + hand-off. Used by the installed app (and mirrors the downloaded app's shell).
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export HOME_DIR="$HOME/Library/Application Support/FilmDeveloper"
export ENGINE="$HOME_DIR/engine"
export LOG="$HOME_DIR/last_run.log"
mkdir -p "$HOME_DIR"
[ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt 2000000 ] && mv -f "$LOG" "$LOG.old"
exec 3>>"$LOG"
echo "==== $(date) engine shell" >&3

note() { osascript -e "display notification \"$1\" with title \"Film Developer\"" >/dev/null 2>&1; }
ver_gt() {
  local IFS=. i; local -a a=($1) b=($2)
  for i in 0 1 2; do
    [ "${a[i]:-0}" -gt "${b[i]:-0}" ] 2>/dev/null && return 0
    [ "${a[i]:-0}" -lt "${b[i]:-0}" ] 2>/dev/null && return 1
  done
  return 1
}
json_get() { plutil -extract "$2" raw -o - "$1" 2>/dev/null; }

REPO="$(json_get "$ENGINE/update.json" repo)"
BRANCH="$(json_get "$ENGINE/update.json" branch)"; BRANCH="${BRANCH:-main}"
SUB="$(json_get "$ENGINE/update.json" path)"; SUB="${SUB:-engine}"
if [ -n "$REPO" ]; then
  # GitHub's API is current immediately; the raw file can be cached for a few minutes (fallback)
  remote_ver="$(curl -fsSL --max-time 6 -H "Accept: application/vnd.github.raw" \
      "https://api.github.com/repos/$REPO/contents/$SUB/VERSION?ref=$BRANCH" 2>/dev/null | tr -d '[:space:]')"
  case "$remote_ver" in [0-9]*.[0-9]*) ;; *)
    remote_ver="$(curl -fsSL --max-time 6 "https://raw.githubusercontent.com/$REPO/$BRANCH/$SUB/VERSION" 2>/dev/null | tr -d '[:space:]')" ;; esac
  local_ver="$(cat "$ENGINE/VERSION" 2>/dev/null || echo 0)"
  if [ -n "$remote_ver" ] && ver_gt "$remote_ver" "$local_ver"; then
    note "Updating to version $remote_ver…"
    printf 'u%s\t-1\tUpdating Film Developer\tDownloading version %s\n' "$$" "$remote_ver" > "$HOME_DIR/progress"
    TMP="$(mktemp -d)"
    if curl -fsSL --max-time 300 "https://codeload.github.com/$REPO/zip/refs/heads/$BRANCH" -o "$TMP/u.zip" >&3 2>&1 \
       && unzip -q "$TMP/u.zip" -d "$TMP" && NEW="$(ls -d "$TMP"/*/"$SUB" 2>/dev/null | head -1)" \
       && [ -f "$NEW/VERSION" ] && [ -f "$NEW/launch.sh" ]; then
      rm -rf "$ENGINE.new" && cp -R "$NEW" "$ENGINE.new" && rm -rf "$ENGINE" && mv "$ENGINE.new" "$ENGINE" \
        && note "Updated to version $remote_ver" && echo "updated to $remote_ver" >&3
    else
      echo "update to $remote_ver failed; keeping $local_ver" >&3
    fi
    rm -rf "$TMP"; rm -f "$HOME_DIR/progress"
  fi
fi
exec /bin/bash "$ENGINE/launch.sh"
