import os
import glob
import arxiv
import pandas as pd
import numpy as np
from bertopic import BERTopic
import sentence_transformers.util, sentence_transformers.SentenceTransformer
import data_utils

class Recommender:
    def __init__(self):
        #* In case we want to initialize later
        pass

    
    def load_library(self,path_to_library,path_to_vectorized,vectorizer):
        """Loads the library from which we will recommend. Included so as to make enlarging the library
        easier if desired. Assumes the library has been vectorized separately.

        Args:
            path_to_library: file path to the original library parquet file.
            path_to_vectorized: file path to the vectorized library (saved as a numpy array).
            vectorizer: object which was used to transform the library text to vectors. Can be any type, e.g.
            a sentence transformer, tfidf scheme, word2vec embedding, etc. Must be able to take in a string
            and return its embedding.
        """
        self.vectorizer = vectorizer
        self.library = pd.read_parquet(path_to_library) #? Do we want to load only certain cols here
        self.vectorized_library = np.loadtxt(path_to_vectorized)

    def load_topic_models(self,path_to_models):
        """Creates a dictionary whose item keys are the file names for each of our topic models and whose
        values are the correspending loaded BERTopic model.
        
        Args:
            path_to_models: file path to the folder containing the topic models.
        """

        #TODO: Adapt to handle other topic models

        model_files = glob.glob(os.path.join(path_to_models,"*"))
        self.topic_models = {f.split("\\")[-1] : BERTopic.load(f) for f in model_files}

        

    def _get_topic_model(self,input):
        #* Placeholder for a function that finds the correct topic model to use for a given input paper.
        pass

    def recommend(self,arxiv_id,use_topics=False):
        """Yields the top 5 most similar articles to an arxiv article input by the user.

        Args:
            arxiv_id: valid arXiv id number.

        Returns:
            Top 5 most similar articles in the library.
        """

        #### FIRST RETRIEVE, CLEAN, AND VECTORIZE INPUT

        #* Retrieve the input metadata
        input = next(arxiv.Search(id_list=[arxiv_id]).results())
        input_metadata = vars(input)
        
        #* Clean the title, abstract, categories, and authors
        input['summary'] = data_utils.clean(input['summary'])
        input['title'] = data_utils.clean(input['title'])
        input['math_categories'] = data_utils.clean_cat_list(input['math_categories'])
        #* clean authors here -- input['authors']

        #* Transform the input abstract
        #! MUST RETURN 2-DIM PYTORCH TENSOR!
        input_sentence = self.vectorizer(input['summary'])
        
        ####

        #### SEARCH LIBRARY FOR TOP 5 MOST SIMILAR ABSTRACTS

        #### NAIVE SEARCH WITHOUT TOPIC INFO
        if not use_topics:
            
            top_5_indices = sentence_transformers.util.semantic_search(
                query_embeddings=input_sentence,
                corpus_embeddings=self.vectorized_library,
                top_k=5
                )
            top_5_papers = self.library.iloc[top_5_indices]
        
            return top_5_papers
        ####

        else:

        #### FIRST FIND THE DOCS IN SIMILAR TOPICS

        #* Select the appropriate topic model for this input. Predict the topic probabilities.
            input_topic_model = self._get_topic_model(input)
            _ , input_probs = input_topic_model.transform(input_sentence)

            topic_indices = np.argpartition(input_probs[0], -3)[-3:] #* Returns the top 3 most likely topics

        #* Retrieve the indicies of the documents contained within these topics

        doc_indicies = self.library.loc[self.library['topic_labels'].isin(topic_indices)].index
        
        #* Retrieve the corresponding embeddings

        embeddings = self.vectorized_library[doc_indicies]    
        ####

        #### NOW SEARCH FOR TOP 5 SIMILAR ABSTRACTS WITHIN THIS SUBSET
        if not use_topics:
            
            top_5_indices = sentence_transformers.util.semantic_search(
                query_embeddings=input_sentence,
                corpus_embeddings=embeddings,
                top_k=5
                )
            top_5_papers = self.library.iloc[top_5_indices]
        
            return top_5_papers
        ####




