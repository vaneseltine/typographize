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

from itertools import chain
from pathlib import Path
import sys

from PIL import Image
import numpy as np


def image_to_array(img):
    return np.array(img.getdata()).reshape((img.height, img.width))


def normalize(image_array):
    """
    For the purpose of comparing monochromatic arrays,
    just bin the values.
    
    ..todo :: 
    
        In the future, probably want to diff from the average,
        play with the contrast, etc. etc.

    """
    bins = np.array([np.average(image_array)])
    return np.digitize(image_array, bins=bins)


def main():
    font_glyph_paths = list(Path("./image/liberation_mono/").glob("*"))
    sample_input_paths = list(Path("./image/handwritten/").glob("*"))
    for sample_input_path in list(sample_input_paths):
        matches = {}
        for font_glyph_path in font_glyph_paths:
            font_glyph = Image.open(font_glyph_path)
            # print(font_glyph_path.relative_to("."), font_glyph, sep="\n")
            sample = Image.open(sample_input_path)
            # print(sample_input_path.relative_to("."), sample, sep="\n")
            if font_glyph.size == sample.size:
                binary_sample = normalize(image_to_array(sample))
                binary_glyph = normalize(image_to_array(font_glyph))
                # print(binary_sample)
                # print(binary_glyph)
                diffs = np.equal(binary_glyph, binary_sample)
                matches[font_glyph_path] = np.sum(diffs)
        print(sample_input_path, *matches.items(), sep="\n    ")


if __name__ == "__main__":
    status = main()
    # sys.exit(status)
