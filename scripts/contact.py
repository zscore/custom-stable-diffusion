#!/usr/bin/env python3
"""Tile images into a labeled contact sheet: contact.py OUT.png img1.png img2.png ..."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

out, paths = sys.argv[1], sys.argv[2:]
cols, tile, pad = 4, 384, 28
rows = (len(paths) + cols - 1) // cols
sheet = Image.new("RGB", (cols * tile, rows * (tile + pad)), "white")
draw = ImageDraw.Draw(sheet)
for i, p in enumerate(paths):
    x, y = (i % cols) * tile, (i // cols) * (tile + pad)
    sheet.paste(Image.open(p).convert("RGB").resize((tile, tile)), (x, y))
    draw.text((x + 6, y + tile + 6), f"{i + 1}. {Path(p).stem.split('-', 2)[-1]}", fill="black")
sheet.save(out)
print(out)
