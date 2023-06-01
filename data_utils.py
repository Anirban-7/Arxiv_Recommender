import pandas as pd
import arxiv
import regex  ## Regular expressions to clean data
import string
from unicodedata import combining, normalize
from sklearn.preprocessing import MultiLabelBinarizer

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

def category_map():
    """Returns a dictonary pairing the encoded arxiv subject categories with their english
    names.
    
    category_map['math.AP'] = 'Analysis of Partial Differential Equations'
    category_map['math.SG'] = 'Symplectic Geometry'

    Returns:
        Dictionary object with keys equal to the list of arxiv subject tags and values their
        english names.
    """
    category_map = {'astro-ph': 'Astrophysics',
    'astro-ph.CO': 'Cosmology and Nongalactic Astrophysics',
    'astro-ph.EP': 'Earth and Planetary Astrophysics',
    'astro-ph.GA': 'Astrophysics of Galaxies',
    'astro-ph.HE': 'High Energy Astrophysical Phenomena',
    'astro-ph.IM': 'Instrumentation and Methods for Astrophysics',
    'astro-ph.SR': 'Solar and Stellar Astrophysics',
    'cond-mat.dis-nn': 'Disordered Systems and Neural Networks',
    'cond-mat.mes-hall': 'Mesoscale and Nanoscale Physics',
    'cond-mat.mtrl-sci': 'Materials Science',
    'cond-mat.other': 'Other Condensed Matter',
    'cond-mat.quant-gas': 'Quantum Gases',
    'cond-mat.soft': 'Soft Condensed Matter',
    'cond-mat.stat-mech': 'Statistical Mechanics',
    'cond-mat.str-el': 'Strongly Correlated Electrons',
    'cond-mat.supr-con': 'Superconductivity',
    'cond-mat': 'Condensed Matter',
    'cs.AI': 'Artificial Intelligence',
    'cs.AR': 'Hardware Architecture',
    'cs.CC': 'Computational Complexity',
    'cs.CE': 'Computational Engineering, Finance, and Science',
    'cs.CG': 'Computational Geometry',
    'cs.CL': 'Computation and Language',
    'cs.CR': 'Cryptography and Security',
    'cs.CV': 'Computer Vision and Pattern Recognition',
    'cs.CY': 'Computers and Society',
    'cs.DB': 'Databases',
    'cs.DC': 'Distributed, Parallel, and Cluster Computing',
    'cs.DL': 'Digital Libraries',
    'cs.DM': 'Discrete Mathematics',
    'cs.DS': 'Data Structures and Algorithms',
    'cs.ET': 'Emerging Technologies',
    'cs.FL': 'Formal Languages and Automata Theory',
    'cs.GL': 'General Literature',
    'cs.GR': 'Graphics',
    'cs.GT': 'Computer Science and Game Theory',
    'cs.HC': 'Human-Computer Interaction',
    'cs.IR': 'Information Retrieval',
    'cs.IT': 'Information Theory',
    'cs.LG': 'Machine Learning',
    'cs.LO': 'Logic in Computer Science',
    'cs.MA': 'Multiagent Systems',
    'cs.MM': 'Multimedia',
    'cs.MS': 'Mathematical Software',
    'cs.NA': 'Numerical Analysis',
    'cs.NE': 'Neural and Evolutionary Computing',
    'cs.NI': 'Networking and Internet Architecture',
    'cs.OH': 'Other Computer Science',
    'cs.OS': 'Operating Systems',
    'cs.PF': 'Performance',
    'cs.PL': 'Programming Languages',
    'cs.RO': 'Robotics',
    'cs.SC': 'Symbolic Computation',
    'cs.SD': 'Sound',
    'cs.SE': 'Software Engineering',
    'cs.SI': 'Social and Information Networks',
    'cs.SY': 'Systems and Control',
    'econ.EM': 'Econometrics',
    'econ.GN': 'General Economics',
    'econ.TH': 'Theoretical Economics',
    'eess.AS': 'Audio and Speech Processing',
    'eess.IV': 'Image and Video Processing',
    'eess.SP': 'Signal Processing',
    'eess.SY': 'Systems and Control',
    'dg-ga': 'Differential Geometry',
    'gr-qc': 'General Relativity and Quantum Cosmology',
    'hep-ex': 'High Energy Physics - Experiment',
    'hep-lat': 'High Energy Physics - Lattice',
    'hep-ph': 'High Energy Physics - Phenomenology',
    'hep-th': 'High Energy Physics - Theory',
    'math.AC': 'Commutative Algebra',
    'math.AG': 'Algebraic Geometry',
    'math.AP': 'Analysis of PDEs',
    'math.AT': 'Algebraic Topology',
    'math.CA': 'Classical Analysis and ODEs',
    'math.CO': 'Combinatorics',
    'math.CT': 'Category Theory',
    'math.CV': 'Complex Variables',
    'math.DG': 'Differential Geometry',
    'math.DS': 'Dynamical Systems',
    'math.FA': 'Functional Analysis',
    'math.GM': 'General Mathematics',
    'math.GN': 'General Topology',
    'math.GR': 'Group Theory',
    'math.GT': 'Geometric Topology',
    'math.HO': 'History and Overview',
    'math.IT': 'Information Theory',
    'math.KT': 'K-Theory and Homology',
    'math.LO': 'Logic',
    'math.MG': 'Metric Geometry',
    'math.MP': 'Mathematical Physics',
    'math.NA': 'Numerical Analysis',
    'math.NT': 'Number Theory',
    'math.OA': 'Operator Algebras',
    'math.OC': 'Optimization and Control',
    'math.PR': 'Probability',
    'math.QA': 'Quantum Algebra',
    'math.RA': 'Rings and Algebras',
    'math.RT': 'Representation Theory',
    'math.SG': 'Symplectic Geometry',
    'math.SP': 'Spectral Theory',
    'math.ST': 'Statistics Theory',
    'math-ph': 'Mathematical Physics',
    'funct-an': 'Functional Analysis',
    'alg-geom': 'Algebraic Geometry',
    'nlin.AO': 'Adaptation and Self-Organizing Systems',
    'chao-dyn': 'Chaotic Dynamics',
    'nlin.CD': 'Chaotic Dynamics',
    'nlin.CG': 'Cellular Automata and Lattice Gases',
    'nlin.PS': 'Pattern Formation and Solitons',
    'nlin.SI': 'Exactly Solvable and Integrable Systems',
    'nucl-ex': 'Nuclear Experiment',
    'nucl-th': 'Nuclear Theory',
    'physics.acc-ph': 'Accelerator Physics',
    'physics.ao-ph': 'Atmospheric and Oceanic Physics',
    'physics.app-ph': 'Applied Physics',
    'physics.atm-clus': 'Atomic and Molecular Clusters',
    'physics.atom-ph': 'Atomic Physics',
    'physics.bio-ph': 'Biological Physics',
    'physics.chem-ph': 'Chemical Physics',
    'physics.class-ph': 'Classical Physics',
    'physics.comp-ph': 'Computational Physics',
    'physics.data-an': 'Data Analysis, Statistics and Probability',
    'physics.ed-ph': 'Physics Education',
    'physics.flu-dyn': 'Fluid Dynamics',
    'physics.gen-ph': 'General Physics',
    'physics.geo-ph': 'Geophysics',
    'physics.hist-ph': 'History and Philosophy of Physics',
    'physics.ins-det': 'Instrumentation and Detectors',
    'physics.med-ph': 'Medical Physics',
    'physics.optics': 'Optics',
    'physics.plasm-ph': 'Plasma Physics',
    'physics.pop-ph': 'Popular Physics',
    'physics.soc-ph': 'Physics and Society',
    'physics.space-ph': 'Space Physics',
    'q-bio.BM': 'Biomolecules',
    'q-bio.CB': 'Cell Behavior',
    'q-bio.GN': 'Genomics',
    'q-bio.MN': 'Molecular Networks',
    'q-bio.NC': 'Neurons and Cognition',
    'q-bio.OT': 'Other Quantitative Biology',
    'q-bio.PE': 'Populations and Evolution',
    'q-bio.QM': 'Quantitative Methods',
    'q-bio.SC': 'Subcellular Processes',
    'q-bio.TO': 'Tissues and Organs',
    'q-fin.CP': 'Computational Finance',
    'q-fin.EC': 'Economics',
    'q-fin.GN': 'General Finance',
    'q-fin.MF': 'Mathematical Finance',
    'q-fin.PM': 'Portfolio Management',
    'q-fin.PR': 'Pricing of Securities',
    'q-fin.RM': 'Risk Management',
    'q-fin.ST': 'Statistical Finance',
    'q-fin.TR': 'Trading and Market Microstructure',
    'quant-ph': 'Quantum Physics',
    'q-alg' : 'Quantum Algebra',
    'stat.AP': 'Applications',
    'stat.CO': 'Computation',
    'stat.ME': 'Methodology',
    'stat.ML': 'Machine Learning',
    'stat.OT': 'Other Statistics',
    'stat.TH': 'Statistics Theory'}
    return category_map


def OHE_cats(df):
    """Return a DataFrame of one-hot-encoded categories of the library with
    the same index as the library
    """
    mlb = MultiLabelBinarizer()
    category_map = category_map()

    def convert_to_eng(cat_array):
        return [category_map['math.' + tag] for tag in cat_array]

    eng_cats = df['strip_cat'].apply(convert_to_eng)
    OHE_array = mlb.fit_transform(eng_cats)
    
    return pd.DataFrame(OHE_array,columns=mlb.classes_,index=df.index)


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



def clean_cats(df):
    """Takes a dataframe of arxiv search results and returns the same dataframe with the
    'categories' columned 'cleaned' in the following sense:
        1. They are returned as a single string separated by commas
        2. All non-math tags are removed
        3. The 'math-ph' tag is replaced by the correct 'math.MP' tag.

    Args:
        df: dataframe of arxiv search results
    """

    

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