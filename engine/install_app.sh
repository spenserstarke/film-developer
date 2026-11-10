#!/bin/bash
# Builds "Film Developer.app" locally (a stay-open AppleScript app) so it has a Dock icon while it
# works. Built on this Mac, it isn't quarantined, so Gatekeeper doesn't block it.
APPLET_VERSION=3
DEST_DIR="/Applications"; [ -w "$DEST_DIR" ] || { DEST_DIR="$HOME/Applications"; mkdir -p "$DEST_DIR"; }
APP="$DEST_DIR/Film Developer.app"
TMP="$(mktemp -d)"
cat > "$TMP/main.applescript" <<'AS'
-- Film Developer app (v3). Runs the engine in the background and shows its progress.
-- The progress window only draws while the run handler is executing, so the loop lives here.
on run
	set base to (POSIX path of (path to home folder)) & "Library/Application Support/FilmDeveloper/"
	set progressFile to base & "progress"
	set logFile to base & "last_run.log"
	set childPID to do shell script "FD_APPLET=1 /bin/bash " & quoted form of (base & "engine/shell.sh") & " >/dev/null 2>&1 & echo $!"
	do shell script "echo \"[applet v3] started engine pid " & childPID & "\" >> " & quoted form of logFile
	set seenToken to ""
	set seenTicks to 0
	set showing to false
	try
		repeat
			try
				do shell script "kill -0 " & childPID
			on error
				exit repeat
			end try
			set p to ""
			try
				set p to do shell script "cat " & quoted form of progressFile & " 2>/dev/null || true"
			end try
			if p is "" then
				set seenToken to ""
				set seenTicks to 0
				if showing then
					set progress total steps to 0
					set progress completed steps to 0
					set progress description to ""
					set progress additional description to ""
					set showing to false
				end if
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
					if seenTicks ≥ 3 or showing then -- only for things still running after ~1 second
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
						if not showing then
							activate -- bring the progress window in front of the dialogs' leftovers
							do shell script "echo \"[applet v3] showing progress: " & (item 3 of parts) & "\" >> " & quoted form of logFile
						end if
						set showing to true
					end if
				end if
			end if
			delay 0.35
		end repeat
	on error errMsg number errNum
		if errNum is -128 then -- Stop button in the progress window
			do shell script "pkill -TERM -P " & childPID & " ; kill " & childPID & " ; rm -f " & quoted form of progressFile & " ; echo '[applet v3] stopped by user' >> " & quoted form of logFile
		else
			do shell script "echo " & quoted form of ("[applet v3] error " & errNum & ": " & errMsg) & " >> " & quoted form of logFile
		end if
	end try
end run
AS
osacompile -o "$TMP/Film Developer.app" "$TMP/main.applescript" || exit 1
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
