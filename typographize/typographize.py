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
from itertools import chain, count
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

CODES = list(i for i in ASCII_CODES_ONLY if chr(i) != " ")
CODES = FULL_CODES


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
    test_image = Image.new(
        mode="1", size=(font_cfg["w"], font_cfg["h"]), color=0
    )
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
    ary = np.array(img_data)
    try:
        reshaped_ary = ary.reshape((img.height, img.width))
    except ValueError:
        print(ary, ary.shape, ary.size, dir(ary))

        sys.exit()
    return reshaped_ary


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


def split_sample(ary, char_height=16, char_width=8, wiggle=0):
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
    overall_height, overall_width = ary.shape
    finite_indices = create_indices(overall_width, char_width, wiggle)
    # split_up = np.split(ary, 8, axis=1)
    # split_indices = list((i + 1) * 8 for i in range(int(overall_width / char_width)))
    # split_indices = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88]
    # print([8, 16, 24, 32, 40, 48, 56])
    # print(*((i + 1) * 8 for i in range(int(64 / 8))))

    split_up = np.array_split(ary, finite_indices, axis=1)
    return split_up


def create_indices(overall_width, char_width, wiggle=0):
    """
    Ugly but effective.

    64, 8, 0:   8, 16, 24, 32, 40, 48, 56

    .. todo:: Wiggle.

    """
    start = char_width + wiggle
    step = char_width
    end = overall_width - char_width
    result = [start]
    while result[-1] < end:
        result.append(result[-1] + step)
    return result


def blaze_samples_matching_and_everything(font_cfg):  # where are you black
    sample_input_paths = list(Path("./image/handwritten/").glob("**/*.*"))
    for sample_input_path in list(sample_input_paths):
        print(sample_input_path)
        sample = Image.open(sample_input_path)
        matches = run_piece_against_fonts(sample, font_cfg)
        if matches:
            winrars = reversed(sorted(matches.items(), key=lambda x: x[1])[-5:])
            winrars = list(
                (chr(i), score * 1.0 / (font_cfg["w"] * font_cfg["h"]))
                for i, score in winrars
            )
            print(*winrars, sep="\n")
            best_of_the_best = sloppily_binarify(
                char=(winrars[0][0]), font_cfg=font_cfg
            )

            blocky_print(normalize(sample), best_of_the_best)


def run_piece_against_fonts(sample, font_cfg):
    try:
        binary_sample = normalize(sample)
    except AttributeError:
        # print("Probably already normalized")
        binary_sample = sample
    matches = {}
    for i in CODES:
        binary_glyph = sloppily_binarify(num=i, font_cfg=font_cfg)
        if binary_glyph.shape != binary_sample.shape:
            print(
                f"bad sizes: "
                f"font {(font_cfg['w'], font_cfg['h'])} "
                f"vs. sample {binary_sample.size}"
            )
            break
        diffs = np.equal(binary_glyph, binary_sample)
        matches[i] = np.sum(diffs)
    return matches


def main():
    """

    .. todo::

        Check the top X or have a tolerance, and pick the most-inked character.
        Maybe even prefer letters.
        I.e., if the best match is - ~ * T ... prefer T.

        Though adding unicode I'm seeing an improvement on that score.

    """
    font = get_font("LiberationMono-Regular", 14)
    # font = get_font("consola", 14)
    font_cfg = load_font_cfg(font)
    blaze_samples_matching_and_everything(font_cfg)

    multicol = Image.open(
        Path("../../downloads/different-blah-74x16.xbm").resolve()
    )
    separated_multicol = split_sample(normalize(multicol))
    winning_chars = []
    for char_sized_piece in separated_multicol:
        matches = run_piece_against_fonts(char_sized_piece, font_cfg)
        if matches:
            winrars = reversed(sorted(matches.items(), key=lambda x: x[1])[-5:])
            winrars = list(
                (
                    f'{chr(i)} {(score * 1.0 / (font_cfg["w"] * font_cfg["h"])):2.2f}'
                )
                for i, score in winrars
            )
            print("\n\n", *winrars, sep="    ")
            best_char = winrars[0][0]
            winning_chars += [best_char]
            blocky_print(
                char_sized_piece,
                sloppily_binarify(char=best_char, font_cfg=font_cfg),
            )
            # blocky_print(char_sized_piece, best_of_the_best)
    blocky_print(normalize(multicol))
    print("".join(winning_chars))


if __name__ == "__main__":
    status = main()
