import pandas as pd, numpy as np
from sklearn.base import BaseEstimator
import re, string
from scipy import sparse
from sklearn.base import BaseEstimator

class NBFeaturer(BaseEstimator):
    def __init__(self, alpha):
        self.alpha = alpha

    def preprocess_x(self, x, r):
        return x.multiply(r)

    def pr(self, x, y_i, y):
        p = x[y==y_i].sum(0)
        return (p+self.alpha) / (p.sum()+self.alpha)

    def fit(self, x, y=None):
        self._r = sparse.csr_matrix(np.log(self.pr(x,1,y) / self.pr(x,0,y)))
        return self

    def transform(self, x):
        x_nb = self.preprocess_x(x, self._r)
        return x_nb

class WordExtractorFeaturer(BaseEstimator):
    def __init__(self, words, empty_doc=''):
        self.words = words
        self.empty_doc = empty_doc
        self.re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
        
    def fit(self, xs, ys=None):
        return self
    
    def transform(self, xs):
        new_xs = []
        for x in xs:
            new_x = ' '.join([tok for tok in self._tokenize(x) if tok in self.words])
            if not new_x:
                new_x = self.empty_doc
            new_xs.append(new_x)
        return new_xs
            
    def _tokenize(self, s): 
        return self.re_tok.sub(r' \1 ', s).split()


class CharGrpExtractorFeaturer(BaseEstimator):
    def __init__(self, char_grps=None):
        self.char_grps = char_grps
        self.re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
        
    def fit(self, xs, ys=None):
        return self
    
    def transform(self, xs):
        new_xs = []
        for x in xs:
            grps = []
            bigrps = []
            for tok in self._tokenize(x):
                grps.extend(extract_group(tok, min_length=3))
                bigrps.extend(extract_ngroup(tok))
                
            new_x = ' '.join([grp for grp in grps if not self.char_grps or grp in self.char_grps])
            new_x2 = ' '.join([grp for grp in bigrps if not self.char_grps or grp in self.char_grps])
            new_xs.append(new_x + ' ' + new_x2)
        return new_xs
            
    def _tokenize(self, s): 
        return self.re_tok.sub(r' \1 ', s).split()


sletters = ['a', 'e', 'i', 'o', 'u', 'y']
def get_char_group(word):
    if not word:
        return ''
    
    if len(word)<=3:
        return word
    
    grp = ''
    #添加所有相连元音
    for i, ch in enumerate(word):
        if ch in sletters:
            grp += ch
        else:
            break
    if len(grp) == len(word):
        return grp
    
    # 添加所有相连辅音，如word=blast,  grp=bl
    starti = len(grp)
    for i in range(starti, len(word)):
        ch = word[i]
        if ch not in sletters:
            grp += ch
        else:
            break
    if len(grp) == len(word):
        return grp
                
    #添加所有元音，如word=blast, grp=bla
    starti = len(grp)
    for i in range(starti, len(word)):
        ch = word[i]
        if ch in sletters:
            grp += ch
        else:
            break
    if len(grp) == len(word):
        return grp
    
    #添加所有相连辅音(除了最后一个)
    starti = len(grp)
    for i in range(starti, len(word)):
        ch = word[i]
        if ch not in sletters:
            grp += ch
        else:
            break
    if len(grp) == len(word):
        return grp
    return grp[:-1]

def extract_group(word, min_length=0):
    groups = []
    while word:
        grp = get_char_group(word)
        if not grp:
            break
        word = word[len(grp):]
        if len(grp) >= min_length:
            groups.append(grp)
    return groups

def extract_ngroup(word, n=2, max_length=100000):
    groups = extract_group(word)
    if len(groups) <n:
        return []
    ngroups = []
    
    for i in range(len(groups)-n+1):
        ngrp =''.join(groups[i:i+n])
        if len(ngrp) < max_length:
            ngroups.append(''.join(groups[i:i+n]))
    return ngroups

if __name__=='__main__':
    f = CharGrpExtractorFeaturer()
    print(f.transform(['this is test omygod']))


