#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''

'''

# here put the import lib

import streamlit as st
import pandas as pd
import numpy as np
import torch
import spacy
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

nlp = spacy.load("en_core_web_sm")
ques = st.text_input('Questions', 'How can I be a good geologist?')
st.text("""Question examples:  
How can I be a good geologist? 
How to speak English fluently? 
What's the most important book that you have ever read?
How do I prevent breast cancer?
What is the step by step guide to invest in share market?""")
doc1 = nlp(ques)
embeddings1 = model.encode(ques, convert_to_tensor=True)
ques_list =['What should I do to be a great geologist?',
            'What makes a good geologist?',
            'What\'s the catch of working as a geologist?',
            'What are the best books you\'ve ever read?',
            'What is the most memorable book you have read that you remember and why?',
            'What is the single most useful book you have ever read?',
            'How can I improve my English speaking fluency?',
            'How can I learn to speak English fluently?',
            'Why can\'t I speak fluently?',
            'What are the best ways to prevent breast cancer?',
            'Is breast cancer preventable?',
            'How can taking care of your body prevent breast cancer?',
            'What is the step by step guide to invest in share market in india?',
            'How To Buy Shares Online In India?',
            'How to invest in stocks for beginners step by step?',
            'How can a beginner invest in share market in India?']

ans = ['' for i in range(len(ques_list))]
similarity_list = list()
print('questions list length:', len(ques_list))
for q in ques_list:
    doc = nlp(q)
    similarity = doc1.similarity(doc)
    similarity_list.append(similarity)

embeddings2 = model.encode(ques_list, convert_to_tensor=True)
cosine_scores = util.cos_sim(embeddings1, embeddings2)
# print(cosine_scores)
# sorted, indices = torch.sort(cosine_scores)
# print(sorted)
# print(indices)

similarity_df = pd.DataFrame({"Questions":ques_list,"Similarity":cosine_scores[0].tolist(), "Answers":ans})
similarity_df.sort_values('Similarity',ascending = False,inplace = True )



if st.button('submit'):
    st.text("Similar questions:")
    st.dataframe(similarity_df.head(5))
    print('end')
