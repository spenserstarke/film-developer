"""Render one photo on every stock: python contact_sheet.py samples/dng_lin.npy out.jpg [width]"""
import sys, glob, pickle, time
import numpy as np
from skimage import transform
from PIL import Image, ImageDraw, ImageFont
import filmsim3 as f3

src, out = sys.argv[1], sys.argv[2]
width = int(sys.argv[3]) if len(sys.argv) > 3 else 700
lin = np.load(src)
H, W = lin.shape[:2]
lin = transform.resize(lin, (int(H * width / W), width), anti_aliasing=True).astype(np.float32)
order = ["kodak_portra_160", "kodak_portra_400", "kodak_portra_800", "kodak_portra_800_push1",
         "kodak_portra_800_push2", "kodak_ektar_100", "kodak_gold_200", "kodak_ultramax_400",
         "kodak_verita_200d", "kodak_vision3_50d", "kodak_vision3_250d", "kodak_vision3_200t",
         "kodak_vision3_500t", "kodak_vision3_500t_no_remjet", "kodak_vision3_50d_no_remjet",
         "fujifilm_c200", "fujifilm_pro_400h", "fujifilm_xtra_400",
         "fujifilm_provia_100f", "fujifilm_velvia_100", "kodak_ektachrome_100"]
tiles = [(f3.linear_to_srgb(lin), "Digital (no film)")]
for k in order:
    st = pickle.load(open(f"stocks/{k}.pkl", "rb"))
    t = time.time()
    img, _ = f3.simulate(st, lin)
    tiles.append((img, st.display_name))
    print(f"{st.display_name}: {time.time() - t:.1f}s", flush=True)
cols = 4
rows = -(-len(tiles) // cols)
h, w = tiles[0][0].shape[:2]
lab = 30
sheet = Image.new("RGB", (cols * (w + 10) + 10, rows * (h + lab + 10) + 10), (24, 24, 24))
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
except OSError:
    font = ImageFont.load_default()
dr = ImageDraw.Draw(sheet)
for n, (img, name) in enumerate(tiles):
    x, y = 10 + (n % cols) * (w + 10), 10 + (n // cols) * (h + lab + 10)
    sheet.paste(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)), (x, y + lab))
    dr.text((x + 2, y + 5), name, fill=(235, 235, 235), font=font)
sheet.save(out, quality=92)
print("saved", out)
