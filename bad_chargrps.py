import pandas as pd, numpy as np
from itertools import chain
from collections import defaultdict
from featurizer import extract_group, extract_ngroup

bws = pd.read_csv('data/badwords.txt', sep='\t')
grp_dict = defaultdict(int)
bigrp_dict = defaultdict(int)
for w in bws['word']:
    w = str(w)
    if w.isdigit():continue
    for grp in extract_group(w):
        if (len(grp)<3):
            continue
        grp_dict[grp] += 1
    for grp in extract_ngroup(w, 2):
        if (len(grp)>7):
            continue
        bigrp_dict[grp] += 1

with open('data/bad_chargrps.txt', 'w') as g:
    g.write('chargrp\tcnt\n')
    for grp, cnt in chain(grp_dict.items(), bigrp_dict.items()):
        if cnt>1:
            g.write('{}\t{}\n'.format(grp, cnt))

