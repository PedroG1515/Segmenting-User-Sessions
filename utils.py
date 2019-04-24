"""Utils functions for session inference."""

import math
import re
from urllib.parse import urlparse
import re
from nltk.tokenize import RegexpTokenizer
###Set###
s = set(['www', 'com'])
tokenizer = RegexpTokenizer(r'\w+')
with open('stoplist.txt') as f:
    content = f.readlines()
stop_words = [x.strip() for x in content]
#########

def distanciaLimite(x0, y0):
    return math.sqrt((x0)**2+(y0)**2)

def filterString(string):
    return re.sub(r'\w+', sub, string)

def extract_url_domain_name(url):
    #FIXME: consider corner case URLs
    domain = urlparse(url).hostname
    return domain.split('.')[1]

def getNgrams(string):
    size = 3
    string = string.strip()
    limit = len(string) - size
    ngrams = []
    for i in range(0, limit+1):
        ngrams.append(string[i:i+size])
    if len(string) >= 4:
        size = 4
        limit = len(string) - size
        for i in range(0, limit+1):
            ngrams.append(string[i:i+size])
    return ngrams and ngrams or [string, ]

def mergeNgrams(bigArray, littleArray):
    return list(set(bigArray).union(set(littleArray)))

def jaccard_similarity(string, ngramsArray):
    stringNgrams = getNgrams(string)
    length = len(stringNgrams)
    intersection = len(set(stringNgrams).intersection(set(ngramsArray)))
    union = (length + len(ngramsArray)) - intersection
    return float(intersection / union)

def sub(m):
    return '' if m.group() in s else m.group()

def size_lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[None]*(n + 1) for i in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]

###GM#####
def overlapping(string, ngramsArray):
    if not ngramsArray and not string:
        return 1
    stringNgrams = get3grams(string)
    length = len(stringNgrams)
    return len(set(stringNgrams).intersection(set(ngramsArray)))/float(length)


def get3grams(string, size=3):
    string = string.strip()
    limit = len(string) - size
    ngrams = []
    for i in range(0, limit+1):
        ngrams.append(string[i:i+size])
    return ngrams and ngrams or [string, ]

def filterStopWords(string):
    string2 = ""
    word_tokens = tokenizer.tokenize(string)
    for w in word_tokens:
        if w not in stop_words:
            string2 = string2 + w + " "
    string2 = string2[:-1]
    return string2