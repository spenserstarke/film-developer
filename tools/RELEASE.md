# Shipping an update
1. Change code in engine/ (or rebuild stocks into engine/stocks/ with tools/build_all.py + patch_stocks.py).
2. Bump engine/VERSION (e.g. 1.0.0 -> 1.0.1). The app only updates when this number increases.
3. Push to the branch named in engine/update.json. Every copy of the app picks it up on next launch.
If engine/requirements*.txt change, the app reinstalls its Python components automatically.
