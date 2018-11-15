#!/usr/bin/env python3

print('hi')

from pathlib import Path

from PIL import Image

test_image_path = Path('./image/o.xbm')
test_image = Image.open(test_image_path)

print(test_image.format, test_image.size, test_image.mode)
print(list(test_image.getdata()))

width, height = test_image.size

for x in range(width):
    for y in range(height):
        coords = (x, y)
        # print(coords, f'{test_image.getpixel(coords):03d}')
        print(f'{test_image.getpixel(coords):03d}', end=' ')
    print('')
