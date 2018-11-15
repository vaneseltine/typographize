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

from pathlib import Path
import sys

from PIL import Image
import numpy


def image_to_array(img):
    return numpy.array(img.getdata()).reshape((img.height, img.width))


def main():
    test_image_path = Path("./image/o.xbm")
    test_image = Image.open(test_image_path)

    print(test_image.format, test_image.size, test_image.mode)
    # print(list(test_image.getdata()))

    # print(dir(test_image))
    width, height = test_image.size

    print(image_to_array(test_image))

    return 0


if __name__ == "__main__":
    status = main()
    # sys.exit(status)
