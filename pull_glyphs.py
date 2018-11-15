#!/usr/bin/env python3

from collections import Counter
from itertools import chain
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont, ImageDraw

INCLUDE_GUIDES = True
AUXILIARY_FONT_DIR = Path("./font/")

CODES = list(
    # liberation eyeballing only:
    # chain(range(32, 128), range(161, 768), range(900, 1155), range(1162, 1282))
    # Further eyeballing with Consolas:
    chain(range(32, 127), range(161, 768), range(900, 1155), range(1162, 1282))
)
ASCII_CODES_ONLY = list(range(32, 127))
# CODES = ASCII_CODES_ONLY


def find_good_glyph_size(font, codes=CODES):
    """
    Find a reasonable size to work with in the provided font and size.
    Currently taking the larger of the top two most commonly used (to
    account for descenders).

    ..todo :: Assess whether it's better to err too large or too small.
    
    """
    # Collect all glyph sizes of the character codes in use
    all_xs = Counter(font.getsize(chr(i))[0] for i in codes)
    all_ys = Counter(font.getsize(chr(i))[1] for i in codes)
    # print(all_xs, all_ys)
    try:
        good_width = max(all_xs.most_common(2)[i][0] for i in range(2))
        good_height = max(all_ys.most_common(2)[i][0] for i in range(2))
    except IndexError:
        good_width = all_xs.most_common(1)[0][0]
        good_height = all_ys.most_common(1)[0][0]
    glyph_size = (good_width, good_height)
    return glyph_size


print("Assessing the following codes:")
for i in range(max(CODES)):
    if i % 50 == 0:
        print("")
    if INCLUDE_GUIDES and i % 10 == 0:
        print(f" {i:04d} ", end="")
    if i in CODES and len(chr(i)) == 1:
        print(f"{chr(i)}", end="")
    else:
        print(" ", end="")
print("\n")

# font_paths = Path("./font").glob("**/*.ttf")
# font_path = next(font_paths)
# with font_path.open("r") as intake:
# font = ImageFont.truetype("./font/LiberationMono-Regular.ttf", 12)
# LiberationMono appears to be 10x13 @ 12
fonts = [
    ("consola.ttf", 12),
    ("consola.ttf", 14),
    ("consola.ttf", 16),
    ("LiberationMono-Regular.ttf", 12),
    ("LiberationMono-Regular.ttf", 14),
    ("LiberationMono-Regular.ttf", 16),
]

for name, size in fonts:
    try:
        font = ImageFont.truetype(name, size)
    except OSError:
        try:
            name = str(Path(AUXILIARY_FONT_DIR, name).relative_to("."))
            font = ImageFont.truetype(name, size)
        except OSError:
            print(f"Cound not find {name}.")
            continue
    print(name, size, find_good_glyph_size(font))
# Consolas 12   9x14
# test_image = Image.new(mode="1", size=(9, 16), color=112)
# draw = ImageDraw.Draw(test_image)


# print(*(numpy.histogram(y) for x, y in font.getsize(chr(i)) for i in CODES))
# for i in CODES:
#    draw.text(xy=(0, 0), text=chr(i), font=font)
#    print(font.getsize(chr(i)))
