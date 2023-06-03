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
import ast
from itertools import chain
 

##Streamlit installation and running instructions###
# Install: pip install streamlit
# Run: streamlit run app.py




# Function to extract the details of the paper
def arxiv_search(input_id):
    paper = next(arxiv.Search(id_list=[input_id]).results())
    return paper

#function to check if one word is contained in another and it returns the compound word
def contained(word1,word2):
    if len(word1)>=len(word2):
        return word1
    else:
        if word1 in word2:
            return word2
        else:
            return word1
        
#function returning the unique sequences/topics
def unique_compound(input_list):
    final_list = []
    location =[]
    
    
    for i in range(len(input_list)):
        for j in range(i+1,len(input_list)):
            
            word = contained(input_list[i],input_list[j])
            if word!= input_list[i]:
                location.append(i)
                
    loc = list(set(location))
    for i in range(len(input_list)):
        if i not in location:
            final_list.append(input_list[i])
    
    return final_list

def topic_recommender(topic_df,location_array):
    all_topics = []
    for i in range(5):
        if topic_df.Unique.iloc[location_array[i]]!='[]':
            all_topics.append(ast.literal_eval(topics.Unique.iloc[location_array[i]]))
#    print(all_topics)       
    if not all_topics:
        return ['Need new input']
        
    return unique_compound(list(set(list(chain.from_iterable(all_topics)))))

if __name__=='__main__':

    st.set_page_config(layout="wide")
#Title for the dashboard
    st.title("ArXiv recommender")
    
    
# Input article; currently only one input
    input_arxiv_id = st.text_input('Insert arXiv id here: ')


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
    
    #Loading the stored corpus and embeddings and topics
    embeddings = pd.read_parquet('./data/df_lib_vecs_20k_all-MiniLM-L6-v2.parquet').values
    topics = pd.read_parquet('./final_data/library_topic_labels_all.parquet')


    #Initializing the model
    model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")

    #Encoding the title and summary of the input article
    input_embedding = model.encode(input_data.summary)

    #Top 5 recommendations from the corpus
    reco = sentence_transformers.util.semantic_search(query_embeddings=input_embedding,
                                         corpus_embeddings=embeddings,top_k=5)

    reco_id = [recs['corpus_id'] for recs in reco[0]]

    # Loading the metadata
    corpus = pd.read_parquet('./final_data/library.parquet')

    st.write("Top 5 similar articles")

    for i in range(5):
        with st.expander("%s"%corpus.title_raw.iloc[reco_id[i]]):
            st.write('Abstract: ',corpus.abstract_raw.iloc[reco_id[i]])
            
    else:
        pass
    
    st.write("Recommended topics: ")
    all_topics = topic_recommender(topics,reco_id)
    topic_embedding = model.encode(all_topics)
    topic_reco = sentence_transformers.util.semantic_search(query_embeddings=input_embedding,
                                         corpus_embeddings=topic_embedding,top_k=5)

    topic_reco_id = [recs['corpus_id'] for recs in topic_reco[0]]
    
    col1,col2,col3,col4,col5 = st.columns(5)
    
    for i in range(1,6):
        with eval('col'+str(i)):
            st.write(all_topics[topic_reco_id[i-1]])
   
    
    


