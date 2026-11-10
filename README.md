# Film Developer

Physically based film simulation: photon-counting grain, published datasheet curves and spectra,
spectral print on real paper data, 22 stocks. Mac app in `Film Developer.app`; this repo is its
update channel.

- `engine/` — what the Mac app runs and auto-updates from (bump `engine/VERSION` to ship).
- `tools/`  — building stocks from data, calibration, validation, demos.

## How updates reach the app
On launch the app reads `engine/update.json` (`repo`, `branch`), fetches `engine/VERSION` from
GitHub, and if it is newer, downloads this repo and replaces its local engine in
`~/Library/Application Support/FilmDeveloper/engine`. Python components are reinstalled only when
`engine/requirements.txt` changes.

## Data credits
Film and paper profiles derived from spektrafilm (Andrea Volpato), CC BY-SA 4.0:
https://github.com/andreavolpato/spektrafilm — derived stock files inherit that licence.
Granularity targets: Kodak / Fujifilm datasheets (see `engine/stocks_table.py`).
