import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="arXix Research article Recommender"
)
st.title("arXix Research Article Recommender")
st.write('''
         Enter some text to get article recommendations! \n
         I have trained a K-Means model using BERT embeddings on the arXiv dataset which is available [here](https://www.kaggle.com/datasets/Cornell-University/arxiv).\n 
         The data set is quite large, so, only a small portion of the dataset was used for training the model.\n
         You can find the code to this application [here](https://github.com/Ketansuhaas/arxiv) ''')
st.sidebar.success("Select a page above")
