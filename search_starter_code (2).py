import streamlit as st
import numpy as np
import numpy.linalg as la
import pickle 

"""
This is a starter code for Assignment 0 of the course, "Hands-on Master Class on LLMs and ChatGPT | Autumn 2023"
taught by Dr. Karthik Mohan.

Computes closest category of a given word or sentence input into a search bar.
The search is implemented through streamlit and can be hosted as a "web app" on the cloud through streamlit as well
Example webpage and search demo: searchdemo.streamlit.app
"""


# Compute Cosine Similarity
def cosine_similarity(x,y):

    x_arr = np.array(x)
    y_arr = np.array(y)

    dot_product = np.dot(x_arr, y_arr)
    norm_a = np.linalg.norm(x_arr)
    norm_b = np.linalg.norm(y_arr)

    if norm_a == 0 or norm_b == 0:
        cos_sim = 0
    else:
        cos_sim = dot_product / (norm_a * norm_b)

    return cos_sim


# Function to Load Glove Embeddings
def load_glove_embeddings(glove_path=r"C:\Users\jp9tr\OneDrive\Masters\Fall23\EE P 500 D - LLM and ChatGPT\glove.6B.50d.txt"):
    """
    First step: Download the 50d Glove embeddings from here - https://www.kaggle.com/datasets/adityajn105/glove6b50d
    Second step: Format the glove embeddings into a dictionary that goes from a word to the 50d embedding.
    Third step: Store the 50d Glove embeddings in a pickle file of a dictionary.
    Now load that pickle file back in this function
    """

    file_location = glove_path

    embedding_dict = {}
    with open(file_location, "r", encoding='utf-8') as f:
        for line in f:
            word = line.split(" ")[0]
            embedding = np.array([float(val) for val in line.split(" ")[1:]])
            embedding_dict[word] = embedding 

    return embedding_dict

# Get Averaged Glove Embedding of a sentence
def averaged_glove_embeddings(sentence, embeddings_dict):
    """
    Simple sentence embedding: Embedding of a sentence is the average of the word embeddings
    """
    words = sentence.split(" ")
    glove_embedding = np.zeros(50)
    count_words = 0

    ############################
    ### WRITE YOUR CODE HERE ###
    ############################

    for word in words:
        if word in embeddings_dict:
            glove_embedding += embeddings_dict[word]
            count_words += 1
    if count_words > 0:
        glove_embedding /= count_words
  
    return glove_embedding
    



# Load glove embeddings
glove_embeddings = load_glove_embeddings()

# Gold standard words to search from
gold_words = ["flower","mountain","tree","car","building"]

# Text Search
st.title("Search Based Retrieval Demo")
st.subheader("Pass in an input word or even a sentence (e.g. jasmine or mount adams)")
text_search = st.text_input("", value="")


# Find closest word to an input word
if text_search:
    input_embedding = averaged_glove_embeddings(text_search, glove_embeddings)
    cosine_sim = {}
    for index in range(len(gold_words)):
        cosine_sim[index] = cosine_similarity(input_embedding, glove_embeddings[gold_words[index]])

    ############################
    ### WRITE YOUR CODE HERE ###
    ############################

    # Sort the cosine similarities
    sorted_indices = np.argsort(list(cosine_sim.values()))[::-1]
    closest_word = gold_words[sorted_indices[0]]


    st.write("(My search uses glove embeddings)")
    st.write("Closest word I have between flower, mountain, tree, car and building for your input is: ")
    st.subheader(closest_word)
    st.write("")
    if closest_word == "tree":
        st.image(r"C:\Users\jp9tr\OneDrive\Masters\Fall23\EE P 500 D - LLM and ChatGPT\leaf.png")
    elif (closest_word == "mountain"):
        st.image(r"C:\Users\jp9tr\OneDrive\Masters\Fall23\EE P 500 D - LLM and ChatGPT\mteverest.jpg")
    elif (closest_word == "flower"):
        st.image(r"C:\Users\jp9tr\OneDrive\Masters\Fall23\EE P 500 D - LLM and ChatGPT\redrose.jpg")
    elif (closest_word == "building"):
        st.image(r"C:\Users\jp9tr\OneDrive\Masters\Fall23\EE P 500 D - LLM and ChatGPT\whitehouse.webp")
    elif (closest_word == "car"):
        st.image(r"C:\Users\jp9tr\OneDrive\Masters\Fall23\EE P 500 D - LLM and ChatGPT\ferrari.jpg")


