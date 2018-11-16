#!/usr/bin/env python3
"""
notes on vscode...
{
    "workbench.colorTheme": "Solarized Dark",
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
}
F5 to run debug
ctrl+1/2/3 to switch back to editor.
"""

from collections import Counter
from itertools import chain
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageFont, ImageDraw

AUXILIARY_FONT_DIR = Path("./font/")

CODES = list(
    # liberation eyeballing only:
    # Further eyeballing with Consolas:
    chain(range(32, 127), range(161, 768), range(900, 1155), range(1162, 1282))
)
ASCII_CODES_ONLY = list(range(32, 127))
# CODES = ASCII_CODES_ONLY


def dump_glyphs():
    for i in range(max(CODES)):
        if i % 40 == 0:
            print("")
        if i % 10 == 0:
            print(f" {i:04d} ", end="")
        if i in CODES:
            print(f"{chr(i)}", end="")
        else:
            print(" ", end="")
    print("\n")


def sloppily_binarify(num, font_cfg):
    test_image = Image.new(
        mode="1", size=(font_cfg["w"], font_cfg["h"]), color=128
    )
    draw = ImageDraw.Draw(test_image)
    draw.text(xy=(0, 0), text=chr(num), font=font_cfg["font"])
    binary_glyph = normalize(image_to_array(test_image))
    return binary_glyph


def load_font_cfg(name="LiberationMono-Regular", pt=14):
    """

    # font_paths = Path("./font").glob("**/*.ttf")
    # font_path = next(font_paths)
    # with font_path.open("r") as intake:
    # font = ImageFont.truetype("./font/LiberationMono-Regular.ttf", 12)
    # LiberationMono appears to be 10x13 @ 12
    fonts = [
        # ("consola.ttf", 12),
        # ("consola.ttf", 14),
        # ("LiberationMono-Regular", 12),
    ]

    for name, size in fonts:
        try:
            font = ImageFont.truetype(name, size)
        except OSError:
            try:
                name = str(Path(AUXILIARY_FONT_DIR, name).relative_to("."))
                font = ImageFont.truetype(name, size)
            except OSError:
                # print(f"Cound not find {name}.")
                continue
    """
    font_cfg = {"name": name, "pt": pt}
    font = ImageFont.truetype(font_cfg["name"], font_cfg["pt"])
    font_cfg["font"] = font
    font_cfg["h"] = font.font.height
    font_cfg["w"] = font.getsize(" ")[0]  # monotype, so arbitary
    return font_cfg
    # test_image = Image.new(mode="1", size=(9, 16), color=112)


def image_to_array(img):
    return np.array(img.getdata()).reshape((img.height, img.width))


def normalize(image_array, inverse=False):
    """
    For the purpose of comparing monochromatic arrays,
    just bin the values.

    ..todo ::

        In the future, probably want to diff from the average,
        play with the contrast, etc. etc.

    """
    if inverse:
        image_array = image_array * -1
    bins = np.array([np.average(image_array)])
    result = np.digitize(image_array, bins=bins)
    return result


def blocky_print(arr):
    blocks = {0: "█", 1: "◦"}
    for i in arr:
        for j in i:
            print(blocks[j], end="")
        print("\n", end="")


def main():
    font_cfg = load_font_cfg()
    # buncha_glyphs = list(
    #    sloppily_binarify(i, font_cfg) for i in ASCII_CODES_ONLY
    # )
    # for x in buncha_glyphs[-4:]:
    #    blocky_print(x)
    INVERSE_INPUT = True

    sample_input_paths = list(Path("./image/handwritten/").glob("**/*.*"))
    for sample_input_path in list(sample_input_paths):
        sample = Image.open(sample_input_path)
        binary_sample = normalize(image_to_array(sample), inverse=INVERSE_INPUT)
        matches = {}
        for i in ASCII_CODES_ONLY:
            if (font_cfg["w"], font_cfg["h"]) != sample.size:
                # print(
                #    f"bad sizes: "
                #    f"font {(font_cfg['w'], font_cfg['h'])} "
                #    f"vs. sample {sample.size}"
                # )
                continue
            binary_glyph = sloppily_binarify(i, font_cfg)
            diffs = np.equal(binary_glyph, binary_sample)
            matches[i] = np.sum(diffs)
        if matches:
            winrars = reversed(sorted(matches.items(), key=lambda x: x[1])[-5:])
            winrars = list((i, chr(i), score) for i, score in winrars)
            print(sample_input_path, *winrars, sep="\n    ")
            best_of_the_best = sloppily_binarify((winrars[0][0]), font_cfg)
            blocky_print(best_of_the_best)
            print("")
            blocky_print(binary_sample)
    # dump_glyphs()


if __name__ == "__main__":
    status = main()
