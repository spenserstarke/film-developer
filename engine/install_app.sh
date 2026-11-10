#!/bin/bash
# Builds "Film Developer.app" locally (a stay-open AppleScript app) so it has a Dock icon while it
# works. Built on this Mac, it isn't quarantined, so Gatekeeper doesn't block it.
APPLET_VERSION=2
DEST_DIR="/Applications"; [ -w "$DEST_DIR" ] || { DEST_DIR="$HOME/Applications"; mkdir -p "$DEST_DIR"; }
APP="$DEST_DIR/Film Developer.app"
TMP="$(mktemp -d)"
cat > "$TMP/main.applescript" <<'AS'
property childPID : ""
property progressFile : ""
property seenToken : ""
property seenTicks : 0
property showing : false

on run
	set base to (POSIX path of (path to home folder)) & "Library/Application Support/FilmDeveloper/"
	set progressFile to base & "progress"
	set childPID to do shell script "FD_APPLET=1 /bin/bash " & quoted form of (base & "engine/shell.sh") & " >/dev/null 2>&1 & echo $!"
end run

on idle
	try
		do shell script "kill -0 " & childPID
	on error
		tell me to quit
		return 1
	end try
	set p to ""
	try
		set p to do shell script "cat " & quoted form of progressFile & " 2>/dev/null || true"
	end try
	if p is "" then
		set seenToken to ""
		set seenTicks to 0
		if showing then hideProgress()
	else
		set AppleScript's text item delimiters to tab
		set parts to text items of p
		set AppleScript's text item delimiters to ""
		if (count of parts) ≥ 4 then
			set tok to item 1 of parts
			if tok is not seenToken then
				set seenToken to tok
				set seenTicks to 0
			end if
			set seenTicks to seenTicks + 1
			-- only show for things still running after about one second
			if seenTicks ≥ 3 or showing then
				set pct to -1
				try
					set pct to (item 2 of parts) as integer
				end try
				if pct < 0 then
					set progress total steps to -1
				else
					set progress total steps to 100
					set progress completed steps to pct
				end if
				set progress description to item 3 of parts
				set progress additional description to item 4 of parts
				set showing to true
			end if
		end if
	end if
	return 0.4
end idle

on hideProgress()
	set progress total steps to 0
	set progress completed steps to 0
	set progress description to ""
	set progress additional description to ""
	set showing to false
end hideProgress

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
