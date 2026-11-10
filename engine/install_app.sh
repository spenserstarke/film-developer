#!/bin/bash
# Builds "Film Developer.app" locally (a stay-open AppleScript app) so it has a Dock icon while it
# works. Built on this Mac, it isn't quarantined, so Gatekeeper doesn't block it.
APPLET_VERSION=1
DEST_DIR="/Applications"; [ -w "$DEST_DIR" ] || { DEST_DIR="$HOME/Applications"; mkdir -p "$DEST_DIR"; }
APP="$DEST_DIR/Film Developer.app"
TMP="$(mktemp -d)"
cat > "$TMP/main.applescript" <<'AS'
property childPID : ""
on run
	set enginePath to (POSIX path of (path to home folder)) & "Library/Application Support/FilmDeveloper/engine/shell.sh"
	set childPID to do shell script "FD_APPLET=1 /bin/bash " & quoted form of enginePath & " >/dev/null 2>&1 & echo $!"
end run
on idle
	try
		do shell script "kill -0 " & childPID
	on error
		tell me to quit
	end try
	return 2
end idle
on reopen
end reopen
AS
osacompile -s -o "$TMP/Film Developer.app" "$TMP/main.applescript" || exit 1
R="$TMP/Film Developer.app/Contents/Resources"
cp "$ENGINE/AppIcon.icns" "$R/applet.icns"
PL="$TMP/Film Developer.app/Contents/Info.plist"
plutil -replace CFBundleIdentifier -string "local.filmdeveloper.app" "$PL"
plutil -replace CFBundleName -string "Film Developer" "$PL"
codesign --force --deep -s - "$TMP/Film Developer.app" >/dev/null 2>&1   # ad-hoc sign (required on Apple Silicon)
rm -rf "$APP" && mv "$TMP/Film Developer.app" "$APP" || exit 1
touch "$APP"
rm -rf "$TMP"
printf '%s\n%s\n' "$APPLET_VERSION" "$APP" > "$HOME_DIR/installed_app"
echo "$APP"
