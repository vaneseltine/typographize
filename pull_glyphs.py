#!/usr/bin/env python3

from itertools import chain
from pathlib import Path

INCLUDE_GUIDES = True

CODES = list(chain(
    range(32, 128),
    range(161, 768),
    range(900, 1155),
    range(1162, 1282),
))
for i in range(max(CODES)):
    if i % 50 == 0:
        print('')
    if INCLUDE_GUIDES and i % 10 == 0:
        print(f' {i:04d} ', end='')
    if i in CODES and len(chr(i)) == 1:
        print(f'{chr(i)}', end='')
    else:
        print(' ', end='')
print('\n')

