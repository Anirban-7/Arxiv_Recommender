import os
import glob
import arxiv
import pandas as pd
import numpy as np
from bertopic import BERTopic
import sentence_transformers.util, sentence_transformers.SentenceTransformer
import data_utils
import pyarrow as pa

class Recommender:
    def __init__(self):
        #* In case we want to initialize later
        pass

    
    def load_library(self,path_to_library,path_to_vectorized,vectorizer,vectorizer_kwargs=None):
        """Loads the library from which we will recommend. Included so as to make enlarging the library
        easier if desired. Assumes the library has been vectorized separately.

        Args:
            path_to_library: file path to the original library parquet file.
            path_to_vectorized: file path to the vectorized library (saved as a numpy array).
            vectorizer: object which was used to transform the library text to vectors. Can be any type, e.g.
            a sentence transformer, tfidf scheme, word2vec embedding, etc. Must be able to take in a string
            and return its embedding.
            vectorizer_kwargs: dictionary consisting of additional keyword arguments to be passed to vectorizer
            when called on an input to the 'recommend' method.
        """
        self.vectorizer = vectorizer
        self.library = pd.read_parquet(path_to_library) #? Do we want to load only certain cols here
        self.vectorized_library = pd.read_parquet(path_to_vectorized).values
        self.vectorizer_kwargs = vectorizer_kwargs

    def load_topic_models(self,path_to_models):
        """Creates a dictionary whose item keys are the file names for each of our topic models and whose
        values are the correspending loaded BERTopic model.

        Also returns the BERTopic model specified by model_name.
        
        Args:
            path_to_models: file path to the folder containing the topic models.
            model_name: the file name of a BERTopic model to return
        """

        #TODO: Adapt to handle other topic models?

        model_files = glob.glob(os.path.join(path_to_models,"*"))
        self.topic_models = {f.split('\\')[-1] : BERTopic.load(f) for f in model_files}
        



    def get_topic_model(self,model_name):
        return self.topic_models[model_name]

    def recommend(self,arxiv_id,use_topics=False):
        """Yields the top 5 most similar articles to an arxiv article input by the user.

        Args:
            arxiv_id: valid arXiv id number.

        Returns:
            Top 5 most similar articles in the library in a pandas dataframe.
        """

        #### FIRST RETRIEVE, CLEAN, AND VECTORIZE INPUT

        #* Retrieve the input metadata
        input = next(arxiv.Search(id_list=[arxiv_id]).results())
        
        #* Clean the title, abstract, categories, and authors
        input_abs = data_utils.clean_data(input.summary)
        input_title = input.title
        input_cats = data_utils.clean_cat_list(input.categories)
        input_authors = [author.name for author in input.authors]
        input_doc = input_title + input_abs

        #* Transform the input abstract
        if self.vectorizer_kwargs:
            input_embedding = self.vectorizer(input_doc + input_abs, **self.vectorizer_kwargs)
        else:
            input_embedding = self.vectorizer(input_doc + input_abs)

        #* VECTORIZER = SENTENCE TRANSFORMER, OUTPUT IS A 1D NP ARRAY
        ####

        #### SEARCH LIBRARY FOR TOP 5 MOST SIMILAR ABSTRACTS

        #### NAIVE SEARCH FOR TOP K=5 MOST SIMILAR WITHOUT TOPIC INFO
        if not use_topics:
            
            recs = sentence_transformers.util.semantic_search(
                query_embeddings=input_embedding,
                corpus_embeddings=self.vectorized_library,
                top_k=5
                )
            rec_inds = [rec['corpus_id'] for rec in recs[0]]
            rec_scores = [rec['score'] for rec in recs[0]]
        
        #### RETURN ROWS OF LIBRARY CORRESPONDING TO RECS WITH SCORES
            rec_df = self.library.iloc[rec_inds]
            rec_df['score'] = pd.Series(rec_scores, index=rec_inds)

            return rec_df
        ####

        else:

        #### FIRST FIND THE DOCS IN SIMILAR TOPICS
        
            #### GET CORRECT TOPIC MODEL
            input_topic_model = self._get_topic_model(input)
            _ , input_probs = input_topic_model.transform(input_embedding)

            #### RETURNS 3 MOST LIKELY TOPICS
            topic_indices = np.argpartition(input_probs[0], -3)[-3:]

            #### GET ALL DOCS IN THESE TOPICS
            doc_indicies = self.library.loc[self.library['topic_labels'].isin(topic_indices)].index
        
            #### GET EMBEDDINGS
            embeddings = self.vectorized_library[doc_indicies]
            ####

        #### NOW SEARCH FOR TOP 5 SIMILAR ABSTRACTS WITHIN THIS SUBSET
            recs = sentence_transformers.util.semantic_search(
            query_embeddings=input_embedding,
            corpus_embeddings=embeddings,
            top_k=5
            )
    
            rec_inds = [rec['corpus_id'] for rec in recs[0] ]
            rec_scores = [rec['score'] for rec in recs[0]]
        
            rec_df = self.library.iloc[rec_inds]
            rec_df['score'] = pd.Series(rec_scores, index=rec_inds)

            return rec_df
        ####


