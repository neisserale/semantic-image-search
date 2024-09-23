import streamlit as st

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle

from PIL import Image
import os
import random

## config page
st.set_page_config(
    page_title='Semantic Image Search',
    layout='wide'
)

#----------------------------------------------------------------------------------------
# Load models
#----------------------------------------------------------------------------------------
@st.cache_resource
def load_data():
    df_features = pd.read_pickle('data/df_features.pkl')
    
    return df_features

#----------------------------------------------------------------------------------------
# Load models
#----------------------------------------------------------------------------------------
@st.cache_resource
def load_knn_model():
    with open('models/knn_model_euclidean.pkl', 'rb') as f:
        knn_model = pickle.load(f)

    return knn_model


## load files
df_features = load_data()
knn_model = load_knn_model()

## get list of painters
painters = df_features['author'].unique()
painters = [p.replace('_', ' ') for p in painters]

#----------------------------------------------------------------------------------------
# StreamLit App
#----------------------------------------------------------------------------------------
## title sidebar
st.sidebar.title("Painters:")

# ## filter painters
search_query = st.sidebar.text_input('Write something', key='search_query')

if search_query:
    painters_filter = [p for p in painters if search_query.lower() in p.lower()]

else:
    painters_filter = painters

## create sidebar with 2 columns
cols = 2
col1, col2 = st.sidebar.columns(cols)
for idx, painter in enumerate(painters_filter):
    if idx % cols == 0:
        with col1:
            ## select a button
            if st.button(painter, key=f"painter_{idx}"):
                st.session_state.painter_selected = painter
                st.session_state.selected_image = None
                st.session_state.painter_selected_changed = True
                st.rerun()

    else:
        with col2:
            ## select a button
            if st.button(painter, key=f"painter_{idx}"):
                st.session_state.painter_selected = painter
                st.session_state.selected_image = None
                st.session_state.painter_selected_changed = True                
                st.rerun()

## title of the app
st.title('Semantic Image Search')

## show images by default
if 'painter_selected' not in st.session_state:
    st.session_state.painter_selected = 'Vincent van Gogh'
    st.session_state.painter_selected_changed = True

## update the image list if the painter has changed
if 'list_img_path_sample' not in st.session_state or st.session_state.painter_selected_changed:
    painter_selected = st.session_state.painter_selected

    ## filter paintings for painter
    p = painter_selected.replace(' ', '_')
    list_img_path = df_features[df_features['author']==p]['img_path'].to_list()

    ## select sample paintings
    sample = 50
    if len(list_img_path) < 50:
        sample = len(list_img_path)

    list_img_path = random.sample(list_img_path, sample)

    ## add list_img_path to a session
    st.session_state.list_img_path_sample = list_img_path
    st.session_state.painter_selected_changed = False  

list_img_path = st.session_state.list_img_path_sample

#----------------------------------------------------------------------------------------
# Semantic Image Search
#----------------------------------------------------------------------------------------
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None

if st.session_state.selected_image is not None:
    ## backbutton for image
    if st.button("Choose other painting"):
        st.session_state.selected_image = None
        st.rerun()

    ## show image
    author = st.session_state.selected_image.split('/')[2].replace('_', ' ')

    img = Image.open(st.session_state.selected_image)
    img = img.resize((150, 200))
    st.image(img, caption=author, width=220)       
    
    ## get image vector        
    feature_vector = df_features[df_features['img_path']==st.session_state.selected_image]
    query = np.array(feature_vector['features'].to_list()).astype('float32')

    ## get top 10 distances
    distances, indices = knn_model.kneighbors(query)
    
    ## get img_path similarities
    similarities = []
    for i, d in zip(indices[0], distances[0]):
        ## get the path_file according the index
        img_path = df_features.loc[i, 'img_path']
        similarities.append([img_path, round(d, 4)])        

    ## show image recommendation
    st.subheader('Top 10 recommendations:')

    cols = 10
    columns = st.columns(cols)
    for idx, similarity in enumerate(similarities):        
        with columns[idx % cols]:
            img_path = similarity[0]
            d = similarity[1]

            author = img_path.split('/')[1].replace('_', ' ')
            author = author + '\n ({})'.format(d) 

            img = Image.open(img_path)
            img = img.resize((150, 200))
            st.image(img, caption=author)

else:
    ## select painter to show
    painter_selected = st.session_state.painter_selected

    ## add subtitle
    st.markdown(f"""
    <h3>Paintings of: <span style='color: #FF4B4B;'>{painter_selected}</span></h3>
    """, unsafe_allow_html=True)

    ## show paintings in columns
    cols = 10
    columns = st.columns(cols)
    for idx, img_path in enumerate(list_img_path):        
        with columns[idx % cols]:
            img = Image.open(img_path)
            img = img.resize((150, 200))
            st.image(img)            
                    
            ## add button for the image
            if st.button(f"Picture {idx + 1}", key=f"picture_{painter_selected}_{idx}"):
                st.session_state.selected_image = img_path                    
                st.rerun()