import pandas as pd
import arxiv
import regex  ## Regular expressions to clean data
import string
from unicodedata import combining, normalize

#### Tools for scraping the arxiv:

def format_query(author='',title='',cat='',abstract=''):
    """Returns a formatted arxiv query string to handle
    simple queries of at most one instance each of these fields.
    To leave a field unspecified, leave the corresponding argument blank.
    
    e.g. format_query(cat='math.AP') will return
    the string used to pull all articles
    with the subject tag 'PDEs',
    since Math.AP is the subject tag
    for 'Analysis of PDEs'.

    Args:
        author: string to search for in the author field.
        title: string to search for in the title field.
        cat: A valid arxiv subject tag. See the full list of these at:
        https://arxiv.org/category_taxonomy
        abstract: string to search for in the abstract field.

    Returns:
        properly formatted query string to return
        all results simultaneously matching all specified fields.
    """

    tags = [f'au:{author}', f'ti:{title}', f'cat:{cat}', f'abs:{abstract}'] 
    # the tag.endswith(':') below
    # is for filtering out tags that
    # we do not pass to the function
    query = ' AND '.join([tag for tag in tags if not tag.endswith(':')])
    return query

def query_to_df(query,max_results):
    """Returns the results of an arxiv API query in a pandas dataframe.

    Args:
        query: string defining an arxiv query
        formatted according to 
        https://info.arxiv.org/help/api/user-manual.html#51-details-of-query-construction
        
        max_results: positive integer specifying
        the maximum number of results returned.

    Returns:
        pandas dataframe with one column for
        indivial piece of metadata of a returned result.
        To see a list of these columns and their descriptions,
        see the documentation for the
        Results class of the arxiv package here:
        http://lukasschwab.me/arxiv.py/index.html#Result

        The 'links' column is dropped. The 'authors' column is replaced by
        a single string containing each author name separated by a comma.
        The 'categories' column is replaced by a single strining containing each
        category tag separated by commas.

    """
    search = arxiv.Search(
            query = query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.LastUpdatedDate
            )
    results = search.results()

    drop_cols = ['authors','links','_raw','categories']
    df = pd.DataFrame()

    for result in results:
        row_dict = {k : v for (k,v) in vars(result).items() if k not in drop_cols}
        row_dict['authors'] = ','.join([author.name for author in result.authors])
        row_dict['categories'] = ','.join([cat for cat in result.categories])
        row = pd.Series(row_dict)
        df = pd.concat([df , row.to_frame().transpose()], axis = 0)

    return df.reset_index(drop=True,inplace=False)

#### Tools for data cleaning:

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
    Return a string of the list of the authors separated by a comma
    where special accents, strokes, and/or letters have been removed. 
    Also, remove periods after abbreviations and hyphens in names.
"""
def clean_authors(data):
    data = regex.sub("-", " ", data)
    data = regex.sub("\.", "", data)
    data = remove_diacritics(data, outliers=str.maketrans(dict(zip(LATIN.split(), ASCII.split()))))
    # If we want to return a list of the author's names
    # data = data.split(',')
    return data

def math_cat_dict():
    """Returns a dictonary pairing the encoded math subject categories with their english
    names. E.g. 
    
    math_cat_dict()['math.AP'] = 'Analysis of PDEs'
    math_cat_dict()['math.SG'] = 'Symplectic Geometry'

    Returns:
        Dictionary object containing the conversion between arxiv math subject tags
        and their english names.
    """
    math_tags = ['AC','AG','AP','AT','CA','CO','CT','CV','DG','DS','FA','GM','GN','GR','GT',
             'HO','IT','KT','LO','MG','MP','NA','NT','OA','OC','PR','QA','RA','RT','SG',
             'SP','ST']

    math_cats = [f'math.{tag}' for tag in math_tags]

    math_subjects = ['Commutative Algebra','Algebraic Geometry','Analysis of PDEs',
                 'Algebraic Topology','Classical Analysis and ODEs','Combinatorics',
                 'Category Theory','Complex Variables','Differential Geometry',
                 'Dynamical Systems','Functional Analysis','General Mathematics',
                 'General Topology','Group Theory','Geometric Topology',
                 'History and Overview','Information Theory','K-Theory and Homology',
                 'Logic','Metric Geometry','Mathematical Physics','Numerical Analysis',
                 'Number Theory','Operator Algebras','Optimization and Control',
                 'Probability','Quantum Algebra','Rings and Algebras',
                 'Representation Theory','Symplectic Geometry','Spectral Theory',
                 'Statistics Theory']

    math_dict = {k : v for (k,v) in zip(math_cats,math_subjects)}
    return math_dict


def clean_cats(df):
    """Takes a dataframe of arxiv search results and returns the same dataframe with the
    'categories' columned 'cleaned' in the following sense:
        1. They are returned as a single string separated by commas
        2. All non-math tags are removed
        3. The 'math-ph' tag is replaced by the correct 'math.MP' tag.

    Args:
        df: dataframe of arxiv search results
    """

    def clean_cat_list(cat_list):
        """Helper function taking a list of arxiv subject tags and returning a list which
        contains only the valid math tags contained in the input list.

        Args:
            cat_list: a list of 'categories' returned by an arxiv query

        Returns:
            the input list with everything that is not a valid math subject category removed,
            and the 'math-ph' mathematical physics tag converted to 'math.MP'.
        """
        out = []
        j = 0
        i = 0

        cats = math_cat_dict()
        
        for cat in cat_list:
            if cat == 'math.MP':
                i+=1
            elif cat in cats.keys():
                out.append(cat)
            elif cat == 'math-ph':
                j+=1
        if i + j > 0:
            out.append('math.MP')
        
        return out


    categories = df['categories']
    split_cats = categories.apply(lambda x: x.split(','))
    df['categories'] = split_cats.apply(clean_cat_list).apply(lambda x: ','.join(x))

    return df

"""
    Return the the lowercase version of the string after removing all 
    newline characters, LaTeX commands, numeric characters, punctuation, 
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