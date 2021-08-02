# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 11:47:40 2021


"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import math
import seaborn as sns


pickle_in = open("next_best_express.pkl","rb")
model=pickle.load(pickle_in)



uploaded_file = st.file_uploader("Choose a XLSX file", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    #df=df.drop(['Unnamed: 0'],axis=1)
    df=df.sort_values(by=['Created Date'])
    
    df['Created Date']=pd.to_datetime(df['Created Date'])
    
    
    
    #Actual_Income=df['Completed Revenue']
    
    
    
    df['weekday'] = df['Created Date'].dt.dayofweek

    
    
    
    date=df['Created Date']
    
    df=df.drop(['Created Date'],axis=1)
    #df=df.drop(['Completed Revenue'],axis=1)
    
    
    
    df['weekday_cos'] = np.cos(2 * np.pi * (df['weekday']/7))
    
    
    
    df=df.drop(['weekday'],axis=1)
    
    df['f1']=np.log(np.exp(df['weekday_cos'])/df['Future Scheduled Jobs'])
    df['f2']=np.abs(np.sqrt(df['tech_count'])-np.log(df['Future Scheduled Jobs']))
    df['f3']=np.abs(np.sqrt(df['Future Scheduled Jobs'])-np.log(df['tech_count']))
    df['f5']=(np.log(df['Future Scheduled Jobs'])-np.abs(df['weekday_cos']))**2
    
    
    
    
    prediction=model.predict(df)
    
    df['pred_Income']=prediction
    
    new_df=pd.DataFrame({'Date':date,'Projected Revenue':prediction})
    
    new_df = new_df.rename(columns={'Date':'index'}).set_index('index')
    
    
    
    st.line_chart(new_df)
    
    

    
   
    st.write('THE PROJECTED REVENUE $ ',round(sum(prediction)))
    #st.write('THE ACTUAL REVENUE $ ',round(sum(Actual_Income)))
    #st.write('THE DIFFERENCE b/w Actual Vs Predicted $ ',round(sum(Actual_Income)-sum(prediction)))



    
    
    
    
    
    
    
    
    
    


    