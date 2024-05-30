#!/usr/bin/env python
# coding: utf-8

# In[315]:


import os
import pathlib

import numpy as np
import pandas as pd


import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_text as text
import streamlit as st
import fitz

MAX_TOKEN=512
bert_tokenizer_params=dict(lower_case=True)

brt_tokenizer=text.BertTokenizer('vocab.txt', **bert_tokenizer_params)

model = tf.saved_model.load('summarizer')
def data_preprocessing(txt):
    txt= tf.constant(txt)[tf.newaxis]
    token_text=brt_tokenizer.tokenize(txt)
    token_text=token_text.merge_dims(-2,-1)
    
    if token_text.bounding_shape.numpy()[1]>MAX_TOKEN:
        inp =  input_adjusting(txt)
        
    else:
        inp =[txt]
        
    return txt
        
def input_adjusting(inp):
    # adjust the input data into smaller exampels according to max input token accepted by model
    lst_of_paras= [p.strip() for p in inp.split('\n') ]
    tokenised_list  = [tf.constant(p)[tf.newaxis] for p in lst_of_paras]

    tokenised_list  = [brt_tokenizer.tokenize(p) for p in tokenised_list]

    tokenised_list  = [p.merge_dims(-2,-1) for p in tokenised_list]

    inp =[]
    
    
    le=0
    st=""
    for p in range(len(tokenised_list)):
        de=le
    
        de+=tokenised_list[p].bounding_shape().numpy()[1]
        stri+=lst_of_paras[p]
          
        if de<MAX_TOKEN:
            le+=tokenised_list[p].bounding_shape().numpy()[1]+1
            st=st.strip()
            st+=" "+lst_of_paras[p]
            flag=True
        else:
            inp.append(st)
            le=0
            st=""
            flag=False
        
    if flag:
        inp.append(st)
    return inp
    

    

def main():
    
    st.title("PDF SUMMARIZER ")

    # Create a file uploader widget
    uploaded_file = st.file_uploader("Upload a pdf file", type=["pdf"])
    if uploaded_file is not None:
        # Process the PDF file here
        pdf_contents = uploaded_file.read()
        pdf_document = fitz.open("pdf", pdf_contents)
        data=""
        for page in pdf_document:
            
            data+=page.get_text('text')
            
            if len(data)>2.5E6:
                st.write("File Contents Grater than2.5MB")
            else:
                
                text = data_preprocessing(data)
        out=""
        for item in text:
            translated_text,x,y = model(tf.constant(item))
            out+="\n"+translated_text
        st.markdown("### Summary")
        st.write(out)

if __name__ == "__main__":
    main()

