import pandas as pd
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')
from nltk.stem import WordNetLemmatizer
import pandas as pd
import joblib
import streamlit as st
import ast

st.set_page_config(
    layout="wide",
    page_title="Recommender"
)
st.title("Recommender")
c1,c2 = st.columns(2)
input_text = st.text_area('Enter text')
@st.cache
def load_model():
    return joblib.load('knn.joblib')
loaded_model = load_model()
prediction = loaded_model.predict(embedder.encode([input_text]))
##########################################################################

titles = pd.read_csv("titles")
ids = pd.read_csv("ids",dtype = str)

titles = titles.iloc[prediction].T
ids = ids.iloc[prediction].T

titles=titles.dropna(how='all')
ids = ids.dropna(how="all")

#st.write(ids.iloc[0,0])
#st.write(ids.iloc[prediction[0],0])
titlesl = []
links = []
z = titles.shape[0]

for i in range(z):
    idd = ids.iloc[i,0]
    if not idd == "" :
        titlesl.append(titles.iloc[i,0])
        url = "https://arxiv.org/abs/" + str(idd)
        links.append(url)
   

df = pd.DataFrame()
df['title'] = titlesl
df['links']=links
df['links'] = df.apply(
    lambda row: '<a href="{}">{}</a>'.format(row['links'], row['links']),
    axis=1)  
st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)    





