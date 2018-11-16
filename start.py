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

FULL_CODES = list(
    # liberation eyeballing only:
    # Further eyeballing with Consolas:
    chain(range(32, 127), range(161, 768), range(900, 1155), range(1162, 1282))
)
ASCII_CODES_ONLY = list(range(32, 127))
TEST_CODES = [ord(x) for x in "IJKLMNOP"]

CODES = ASCII_CODES_ONLY


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


def sloppily_binarify(num=None, char=None, font_cfg=None):
    stringed_input = char or chr(num)
    test_image = Image.new(mode="1", size=(font_cfg["w"], font_cfg["h"]), color=0)
    draw = ImageDraw.Draw(test_image)
    draw.text(xy=(0, 0), text=stringed_input, font=font_cfg["font"], fill=1)
    binary_glyph = normalize(test_image)
    return binary_glyph


def get_font(font, size):
    attempts = [
        font,
        Path(AUXILIARY_FONT_DIR, font),
        Path(AUXILIARY_FONT_DIR, font + ".ttf"),
        font.lower(),
    ]
    for x in attempts:
        try:
            font = ImageFont.truetype(font=str(x), size=size)
            print(f"Located {x}")
            return font
        except OSError:
            print(f"Could not find {x}")
            continue
    raise FileNotFoundError


def load_font_cfg(font):
    """
    name="LiberationMono-Regular", pt=14):

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
    # font_cfg = {"name": name, "pt": pt}
    font_cfg = {
        "font": font,
        "h": font.font.height,
        "w": font.getsize(" ")[0],  # monotype, so arbitary
    }
    return font_cfg
    # test_image = Image.new(mode="1", size=(9, 16), color=112)


def image_to_array(img):
    img_data = img.getdata()
    return np.array(img_data).reshape((img.height, img.width))


def normalize(img):
    """
    For the purpose of comparing monochromatic arrays,
    just bin the values.

    ..todo ::

        In the future, probably want to diff from the average,
        play with the contrast, etc. etc.

    """
    image_array = image_to_array(img)
    bins = np.array([np.average(image_array)])
    result = np.digitize(image_array, bins=bins)
    return result


def blocky_print(*arrays):
    """
    Print out arrays as blocks to the screen, side-by-side.

    .. todo:: There must surely be a better way to map str dict onto binary arr
    """
    combined_arr = np.concatenate([*arrays], axis=1)
    blocks = {0: "█", 1: " "}  # ◦
    for i in combined_arr:
        for j in i:
            print(blocks[j], end="")
        print("\n", end="")


def split_sample(ary):
    """
    https://docs.scipy.org/doc/numpy-1.10.4/reference/generated/numpy.split.html#numpy.split
    Parameters:
    ary : ndarray
        Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D array
        If indices_or_sections is an integer, N,
            the array will be divided into N equal arrays along axis.
            If such a split is not possible, an error is raised.
        If indices_or_sections is a 1-D array of sorted integers,
            the entries indicate where along axis the array is split.
            For example, [2, 3] would, for axis=0, result in
                ary[:2]
                ary[2:3]
                ary[3:]
        If an index exceeds the dimension of the array along axis,
        an empty sub-array is returned correspondingly.
    axis : int, optional
        The axis along which to split, default is 0.
    Returns:
    sub-arrays : list of ndarrays
        A list of sub-arrays.
    """
    # print((ary - 1) * -1)
    split_up = np.split(ary, 8, axis=1)
    for x in (ary, split_up[0]):
        print(type(x), x.shape, x.dtype)
    blocky_print(*split_up)
    for x in split_up:
        blocky_print(x)


def matching_and_everything(font_cfg):  # where are you black
    sample_input_paths = list(Path("./image/handwritten/").glob("**/*.*"))
    for sample_input_path in list(sample_input_paths):
        sample = Image.open(sample_input_path)
        binary_sample = normalize(sample)
        matches = {}
        for i in CODES:
            if (font_cfg["w"], font_cfg["h"]) != sample.size:
                print(
                    f"bad sizes: "
                    f"font {(font_cfg['w'], font_cfg['h'])} "
                    f"vs. sample {sample.size}"
                )
                break
            binary_glyph = sloppily_binarify(num=i, font_cfg=font_cfg)
            diffs = np.equal(binary_glyph, binary_sample)
            matches[i] = np.sum(diffs)
        if matches:
            winrars = reversed(sorted(matches.items(), key=lambda x: x[1])[-5:])
            winrars = list(
                (chr(i), score * 1.0 / (font_cfg["w"] * font_cfg["h"]))
                for i, score in winrars
            )
            print(sample_input_path, *winrars, sep="\n    ")
            best_of_the_best = sloppily_binarify(
                char=(winrars[0][0]), font_cfg=font_cfg
            )

            blocky_print(binary_sample, best_of_the_best)


def main():
    # font = get_font("LiberationMono-Regular", 14)
    font = get_font("consola", 14)
    font_cfg = load_font_cfg(font)
    matching_and_everything(font_cfg)
    multicol = Image.open(Path("C:/Users/matvan/", "Downloads/different.bmp").resolve())
    separated_multicol = split_sample(normalize(multicol))
    blocky_print(normalize(multicol))


if __name__ == "__main__":
    status = main()
