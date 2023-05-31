import streamlit as st
import os
import glob
import arxiv
import pandas as pd
import numpy as np
from bertopic import BERTopic
import sentence_transformers.util, sentence_transformers.SentenceTransformer
import data_utils
import pyarrow as pa
import subprocess
import sys

##Streamlit installation and running instructions###
# Install: pip install streamlit
# Run: streamlit run app.py




#Title for the dashboard
st.title("ArXiv recommender")

# Input article; currently only one input
input_arxiv_id = st.text_input('Insert arXiv id here: ')

# Function to extract the details of the paper
def arxiv_search(input_id):
    paper = next(arxiv.Search(id_list=[input_id]).results())
    return paper

if input_arxiv_id:
    #Details of the extracted paper are stored
    input_data = arxiv_search(input_arxiv_id)

    #if st.button('Show Abstract'):
    #    st.write('Abstract: ',data.summary)
    #else:
    #    pass 

    # Dropdown for the input article
    with st.expander("%s"%input_data.title):
        st.write('Abstract: ',input_data.summary)
    
    #Loading the stored corpus embeddings
    embeddings = pd.read_parquet('./data/df_lib_vecs_20k_sbert.parquet').values


    #Initializing the model
    model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")

    #Encoding the title and summary of the input article
    input_embedding = model.encode(input_data.summary)

    #Top 5 recommendations from the corpus
    reco = sentence_transformers.util.semantic_search(query_embeddings=input_embedding,
                                         corpus_embeddings=embeddings,top_k=5)

    reco_id = [recs['corpus_id'] for recs in reco[0]]

    # Loading the metadata
    corpus = pd.read_parquet('./data/filter_20k.parquet')

    st.write("Top 5 similar articles")

    for i in range(5):
        with st.expander("%s"%corpus.title.iloc[reco_id[i]]):
            st.write('Abstract: ',corpus.abstract.iloc[reco_id[i]])
            
    else:
        pass





   