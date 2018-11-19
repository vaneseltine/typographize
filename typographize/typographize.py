#!/usr/bin/env python3
"""
F5 to run debug
ctrl+1/2/3 to switch back to editor.
"""

from collections import Counter
from itertools import chain, count
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageFont, ImageDraw


FULL_CODES = list(
    # Further eyeballing with Consolas:
    chain(range(32, 127), range(161, 768), range(900, 1155), range(1162, 1282))
)
ASCII_CODES_ONLY = list(range(32, 127))

# CODES = list(i for i in ASCII_CODES_ONLY if chr(i) != " ")
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


def sloppily_binarify(font, num=None, char=None):
    stringed_input = char or chr(num)
    test_image = Image.new(mode="1", size=(font.width, font.height), color=1)
    draw = ImageDraw.Draw(test_image)
    draw.text(xy=(0, 0), text=stringed_input, font=font.image_font, fill=0)
    binary_glyph = normalize(test_image)
    return binary_glyph


def normalize(image_array):
    """
    For the purpose of comparing monochromatic arrays,
    just bin the values.

    ..todo ::

        In the future, probably want to diff from the average,
        play with the contrast, etc. etc.

    """
    # image_array = image_to_array(img)
    bins = np.array([np.average(image_array)])
    result = np.digitize(image_array, bins=bins)
    return result


def blocky_print(*arrays):
    """
    Print out arrays as blocks to the screen, side-by-side.

    .. todo:: There must surely be a better way to map str dict onto binary arr
    """
    combined_arr = np.concatenate([*arrays], axis=1)
    blocks = {0: " ", 1: "█"}  # ◦
    for i in combined_arr:
        for j in i:
            print(blocks[j], end="")
        print("\n", end="")


def split_sample(ary, char_height, char_width, offset):
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
    finite_indices = create_slice_indices(overall_width, char_width, offset)
    # split_up = np.split(ary, 8, axis=1)
    # split_indices = list((i + 1) * 8 for i in range(int(overall_width / char_width)))
    # split_indices = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88]
    # print([8, 16, 24, 32, 40, 48, 56])
    # print(*((i + 1) * 8 for i in range(int(64 / 8))))

    split_up = np.array_split(ary, finite_indices, axis=1)
    return split_up


def create_slice_indices(overall_width, char_width, offset=0):
    """
    .. todo:: There is an unnecessary 0 that could be rid of.
    """
    offset = offset % char_width
    return list(range(offset, overall_width, char_width))


def blaze_samples_matching_and_everything(font):
    # sample_input_paths = list(Path("./image/handwritten/").glob("**/*.*"))
    sample_input_paths = list(Path("./image/xbm/").glob("**/*.*"))
    for sample_input_path in list(sample_input_paths):
        print(sample_input_path)
        sample = Image.open(sample_input_path)
        matches = run_piece_against_fonts(sample, font)
        if matches:
            winrars = reversed(sorted(matches.items(), key=lambda x: x[1])[-5:])
            winrars = list(
                (chr(i), score * 1.0 / (font.width * font.height))
                for i, score in winrars
            )
            print(*winrars, sep="\n")
            best_of_the_best = sloppily_binarify(char=(winrars[0][0]), font=font)

            blocky_print(normalize(sample), best_of_the_best)


class TypoImage:
    def __init__(self, filepath):
        self.full_image = Image.open(filepath)
        self.data = self.full_image.getdata()
        self.height = self.full_image.height
        self.width = self.full_image.width
        self.full_array = np.array(self.data).reshape(self.height, self.width)
        self.normalized = normalize(self.full_array)
        self.parts = None

    def split(self, char_height, char_width, offset=0):
        """
        https://docs.scipy.org/doc/numpy-1.10.4/reference/generated/numpy.split.html#numpy.split
        """
        y_indices = create_slice_indices(self.height, char_height, offset)
        split_y = np.array_split(self.normalized, y_indices, axis=0)
        print(split_y)
        sys.exit()
        x_indices = create_slice_indices(self.width, char_width, offset)
        split_x = np.array_split(self.normalized, x_indices, axis=1)
        print(split_x)
        sys.exit()

    def slicer(self, chunk_i, chunk_j):
        """
        https://stackoverflow.com/questions/41214432/
        """
        a = self.normalized
        n = self.cover_multiple(a.shape[0], chunk_i)
        m = self.cover_multiple(a.shape[1], chunk_j)
        c = np.empty((n, m))
        c.fill(np.nan)
        c[: a.shape[0], : a.shape[1]] = a
        c = c.reshape(n // chunk_i, chunk_i, m // chunk_j, chunk_j)
        c = c.transpose(0, 2, 1, 3)
        return c

    def cover_multiple(self, current_length, multiple):
        return ((current_length - 1) // multiple + 1) * multiple

    def print(self):
        """
        Print out arrays as blocks to the screen, side-by-side.

        .. todo:: There must surely be a better way to map str dict onto binary arr
        https://stackoverflow.com/questions/16992713/

        """
        blocks = {0: "█", 1: " "}  # ◦
        for i in self.normalized:
            for j in i:
                print(blocks[j], end="")
            print("\n", end="")


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


def image_to_array(img):
    ary = np.array(self.data)
    try:
        reshaped_ary = ary.reshape((img.height, img.width))
    except ValueError:
        print(ary, ary.shape, ary.size, dir(ary))
        sys.exit()
    return reshaped_ary


class Font:

    AUXILIARY_FONT_DIR = Path("./font/")

    def __init__(self, name, pt, codes):
        self.name = name
        self.pt = pt
        self.codes = codes

        self.image_font = self.get_font(name, pt)
        self.height = self.image_font.font.height
        arbitary_test_char = " "
        self.width = self.image_font.getsize(arbitary_test_char)[0]
        self.shape = (self.height, self.width)
        self.load_glyphs()

    def load_glyphs(self):
        self.glyphs = {code: self.sloppily_binarify(code) for code in self.codes}

    def sloppily_binarify(self, code):
        glyph = chr(code)
        test_image = Image.new(mode="1", size=(self.width, self.height), color=1)
        draw = ImageDraw.Draw(test_image)
        draw.text(xy=(0, 0), text=glyph, font=self.image_font, fill=0)
        binary_glyph = normalize(test_image)
        return binary_glyph

    def get_font(self, name, size):
        """
        # test_image = Image.new(mode="1", size=(9, 16), color=112)
        """
        potential_locations = [
            name,
            name.lower(),
            Path(self.AUXILIARY_FONT_DIR, name),
            Path(self.AUXILIARY_FONT_DIR, name + ".ttf"),
        ]
        for loc in potential_locations:
            try:
                font = ImageFont.truetype(font=str(loc), size=size)
                print(f"Located {loc}")
                return font
            except OSError:
                continue
        raise FileNotFoundError(f"Could not find {loc}")

    def match_sample_piece(self, piece):
        total_ink = sum(sum(piece))
        if np.isnan(total_ink):
            # print("Incomplete block.")
            return []
        try:
            binary_sample = normalize(piece)
        except AttributeError:
            print("Probably already normalized")
            binary_sample = piece
        if self.shape != piece.shape:
            print(
                (
                    f"Piece size {binary_sample.shape} "
                    + f"does not match font size {self.shape}"
                )
            )
        # print(binary_sample)
        matches = {}
        for code, binary_glyph in self.glyphs.items():
            diffs = np.equal(binary_glyph, binary_sample)
            matches[code] = np.sum(diffs)
        return matches


def run_piece_against_fonts(sample, font):
    total_ink = sum(sum(char_sized_piece))
    if np.isnan(total_ink):
        # print("Incomplete block.")
        return []
    try:
        binary_sample = normalize(sample)
    except AttributeError:
        # print("Probably already normalized")
        binary_sample = sample
    matches = {}
    for code, binary_glyph in CODES.items():
        diffs = np.equal(binary_glyph, binary_sample)
        matches[code] = np.sum(diffs)
    return matches


def main():
    """

    .. todo::

        Check the top X or have a tolerance, and pick the most-inked character.
        Maybe even prefer letters.
        I.e., if the best match is - ~ * T ... prefer T.

        Though adding unicode I'm seeing an improvement on that score.

    """
    # font = get_font("LiberationMono-Regular", 14)
    # font = get_font("consola", 14)
    match_font = Font(name="LiberationMono-Regular", pt=6, codes=ASCII_CODES_ONLY)
    # blaze_samples_matching_and_everything(font)
    # multicol_img_paths = Path("./image/xbm/").glob("*x35.xbm")
    # multicol_img_paths = Path("./image/xbm/").glob("bigf*.xbm")
    # multicol_img_paths = Path("./image/xbm/").glob("*16.xbm")
    # for img_path in multicol_img_paths:
    # img_path = Path("./image/xbm/different-blah-80x16.xbm")
    # img_path = Path("./image/xbm/whiteonblack-80x16.xbm")
    # img_path = Path("./image/xbm/whiteonblack-74x32.xbm")
    # img_path = Path("./image/xbm/whiteonblack-75x33.xbm")
    # img_path = Path("./image/xbm/sun.xbm")
    # img_path = Path("./image/xbm/tuxedo.xbm")
    # img_path = Path("./image/xbm/bigface.xbm")
    img_path = Path("./image/wdot.pbm")
    multicol = TypoImage(img_path)
    print("\n", img_path)
    chopped_up = multicol.slicer(match_font.height, match_font.width)
    grid = (multicol.height // match_font.height, multicol.width // match_font.width)
    total_chars = grid[0] * grid[1]
    winning_chars = []
    for row in chopped_up:
        for char_sized_piece in row:
            matches = match_font.match_sample_piece(char_sized_piece)
            if matches:
                winrars = reversed(sorted(matches.items(), key=lambda x: x[1])[-10:])
                winrars = list(
                    (
                        chr(i)
                        + f"{(score * 1.0 / (match_font.width * match_font.height)):2.2f}"
                    )
                    for i, score in winrars
                )
                print(
                    *winrars,
                    len([i for i in winning_chars if i is not "\n"]),
                    "of",
                    f"{total_chars}ish",
                    sep="    ",
                )
                best_char = winrars[0][0]
                winning_chars += [best_char]
                blocky_print(
                    char_sized_piece, sloppily_binarify(char=best_char, font=match_font)
                )
                # if best_char == "@":
                #    sys.exit()
        winning_chars += "\n"
    # multicol.print()
    print("".join(winning_chars))


if __name__ == "__main__":
    status = main()

