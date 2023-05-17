import regex  ## Regular expressions to clean data
import string
from unicodedata import combining, normalize

"""
    Return the string, by "removing" any LaTeX code
"""


def remove_latex(data):
    ## First, get rid of any LaTex expressions of the form \cite{...}, \ref{...}, etc..
    data = regex.sub("\\\cite\{.*?\}", '', data)

    ## Then, get rid of math code $...$, $$...$$, etc...
    data = regex.sub("\$+.*?\$+", '', data)

    return data


"""
    Return the string, by "removing" any
    diacritics like accents or curls and strokes (as in ø etc.) and the like.

    Reference: a blog post from
    https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string
"""

LATIN = "ä  æ  ǽ  đ ð ƒ ħ ı ł ø ǿ ö  œ  ß  ŧ ü "
ASCII = "ae ae ae d d f h i l o o oe oe ss t ue"


def remove_diacritics(string, outliers=str.maketrans(dict(zip(LATIN.split(), ASCII.split())))):
    return "".join(c for c in normalize("NFD", string.lower().translate(outliers)) if not combining(c))


"""
    Return the the lowercase version of the string after removing all 
    newline characters, LaTeX commands, numeric characters, puntuation, 
    and any accents, strokes, special characters in names.
"""


def clean_data(data):
    ## Convert to lowercase
    data = data.lower()
    ## Get rid of newline character. Do this before removing LaTex code,
    ## otherwise remove_latex may not remove everything.
    data = regex.sub(r"\n", " ", data)
    data = remove_latex(data)

    ## Remove hyphens before removing other puntuation so we retain names like
    ## Navier Stokes as separate words.
    data = regex.sub("-", " ", data)

    data = regex.sub('[0-9]+', '', data)

    data = " ".join(data.split())

    ## Get rid of any remaining puntuation.
    ## string.punctuation is a String of ASCII characters which are considered punctuation
    ## characters in the C locale: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.
    data = "".join([char for char in data if char not in string.punctuation])

    ## Remove accents in words like schrödinger, montréal, über, 12.89, mère, noël, etc
    data = remove_diacritics(data, outliers=str.maketrans(dict(zip(LATIN.split(), ASCII.split()))))

    return data