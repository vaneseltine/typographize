#!/usr/bin/env python3

import sys

import pytest

# sys.path.append("./../")
# sys.path.append("./../typographize/")

import typographize.typographize as ty

print(dir(ty))


@pytest.mark.parametrize(
    "input, expected",
    [
        ((62, 8, -1), [7, 15, 23, 31, 39, 47, 55]),
        ((63, 8, -1), [7, 15, 23, 31, 39, 47, 55]),
        ((64, 8, -1), [7, 15, 23, 31, 39, 47, 55, 63]),
        ((65, 8, -1), [7, 15, 23, 31, 39, 47, 55, 63]),
        ((66, 8, -1), [7, 15, 23, 31, 39, 47, 55, 63]),
        ((62, 8, 0), [8, 16, 24, 32, 40, 48, 56]),
        ((63, 8, 0), [8, 16, 24, 32, 40, 48, 56]),
        ((64, 8, 0), [8, 16, 24, 32, 40, 48, 56]),
        ((65, 8, 0), [8, 16, 24, 32, 40, 48, 56, 64]),
        ((62, 8, 1), [9, 17, 25, 33, 41, 49, 57]),
        ((63, 8, 1), [9, 17, 25, 33, 41, 49, 57]),
        ((64, 8, 1), [9, 17, 25, 33, 41, 49, 57]),
        ((65, 8, 1), [9, 17, 25, 33, 41, 49, 57]),
        ((66, 8, 1), [9, 17, 25, 33, 41, 49, 57, 65]),
    ],
)
def test_create_indices_simple(input, expected):
    overall_width, char_width, wiggle = input
    assert ty.create_indices(overall_width, char_width, wiggle) == expected


if __name__ == "__main__":
    pytest.main(sys.argv)
