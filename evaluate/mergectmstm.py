#!/usr/bin/env python

import sys

ctmFile=sys.argv[1]
stmFile=sys.argv[2]

with open(ctmFile, 'r') as f:
    lines = [line.strip() for line in f.readlines()]
    ctmDict_raw = [line.split()[0] for line in lines]
    ctmDict = list(set(ctmDict_raw))

with open(stmFile, 'r') as f:
    stmDict = [line.strip().split()[0] for line in f.readlines()]

empty_list = [item for item in stmDict if not (item in ctmDict)]


for item in empty_list:
    k = stmDict.index(item)
    if k == len(stmDict) - 1:
        lines.append('%s 1 0.000 0.030 [EMPTY]' % item)
    else:
        item_next = stmDict[k + 1]
        idx = ctmDict_raw.index(item_next)
        lines.insert(idx, '%s 1 0.000 0.030 [EMPTY]' % item)

    ctmDict_raw = [line.split()[0] for line in lines]


with open(ctmFile, 'w+') as f:
    for line in lines:
        f.write(line + '\n')
