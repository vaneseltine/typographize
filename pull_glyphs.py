#!/usr/bin/env python3

from collections import Counter
from itertools import chain
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont, ImageDraw

INCLUDE_GUIDES = True

CODES = list(
    # liberation eyeballing only:
    # chain(range(32, 128), range(161, 768), range(900, 1155), range(1162, 1282))
    chain(range(32, 128), range(161, 768), range(900, 1155), range(1162, 1282))
)


def find_good_glyph_size(font):
    all_xs = Counter(font.getsize(chr(i))[0] for i in CODES)
    all_ys = Counter(font.getsize(chr(i))[1] for i in CODES)
    # print(all_xs, all_ys)
    good_width = max(all_xs.most_common(2)[i][0] for i in range(2))
    good_height = max(all_ys.most_common(2)[i][0] for i in range(2))
    glyph_size = (good_width, good_height)
    return glyph_size


# for i in range(max(CODES)):
#    if i % 50 == 0:
#        print("")
#    if INCLUDE_GUIDES and i % 10 == 0:
#        print(f" {i:04d} ", end="")
#    if i in CODES and len(chr(i)) == 1:
#        print(f"{chr(i)}", end="")
#    else:
#        print("X", end="")
# print("\n")

# font_paths = Path("./font").glob("**/*.ttf")
# font_path = next(font_paths)
# with font_path.open("r") as intake:
# font = ImageFont.truetype("./font/LiberationMono-Regular.ttf", 12)
# LiberationMono appears to be 10x13 @ 12
fonts = [
    ("C:/Windows/Font/consola.ttf", 12),
    ("C:/Windows/Font/consola.ttf", 14),
    ("C:/Windows/Font/consola.ttf", 16),
    ("./font/LiberationMono-Regular.ttf", 12),
    ("./font/LiberationMono-Regular.ttf", 14),
    ("./font/LiberationMono-Regular.ttf", 16),
]

for name, size in fonts:
    font = ImageFont.truetype(name, size)
    print(name, size, find_good_glyph_size(font))
# Consolas 12   9x14
# test_image = Image.new(mode="1", size=(9, 16), color=112)
# draw = ImageDraw.Draw(test_image)


# print(*(numpy.histogram(y) for x, y in font.getsize(chr(i)) for i in CODES))
# for i in CODES:
#    draw.text(xy=(0, 0), text=chr(i), font=font)
#    print(font.getsize(chr(i)))
