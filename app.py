import numpy as np
import pandas as pd
import pickle
import time
import json
import streamlit as st
from streamlit_lottie import st_lottie
import plotly.express as px

st.set_page_config(page_title="Bengaluru House price predictor",page_icon="icons/house_logo.png",layout="wide")
st.sidebar.subheader("House Price Predictor")
st.sidebar.image("icons/house_logo.png",width=150)
st.subheader("Welcome! to House Price Predictor💰")
df = pd.read_csv("cleaned_df.csv")

selection = st.segmented_control(None,['Home','Predict Price',"Dashboard",'Sample Data'],default='Home')

def load_house_anime():
    with open("home_anime.json",'rb') as file:
        anime = json.load(file)
        return anime

if selection=='Home':
    c1,c2=st.columns([1,2])
    with c1:
        anime = load_house_anime()
        st_lottie(anime,width=300)
    with c2.container(border=True,height=300):
        st.markdown("""
                    ##### 🏠 About This App

    Welcome to the **House Price Prediction App**!  
    This tool helps you estimate the price of a house based on key features like location, square footage, number of bedrooms, and number of bathrooms.

    ###### 📍 Dataset Used:
    The predictions and charts are based on real data from the **Bangalore House Price dataset**, which contains detailed information about residential properties in Bangalore.

    ##### ✨ What You Can Do:
    - **📊 Dashboard:** Explore charts and visualizations to understand Bangalore’s house price trends.
    - **💰 Price Prediction:** Get an estimated price for your property based on your inputs.
    - **🗂️ Sample Data:** View a sample of the Bangalore housing dataset.

    This app is built with **Streamlit** and powered by a Machine Learning model trained on Bangalore housing market data. Use it for learning, exploration, or quick insights — but always consult real market experts for final decisions!

    Happy predicting! 🏡✨

                    """)
elif selection=="Predict Price":
    with st.container(border=True):
        col1,col2 = st.columns(2)
        loc = col1.selectbox("📍Location: ",options=df['location'].unique())
        sqft = col1.selectbox("📐Total Sqft: ",options=np.arange(300.0,35000.0,100.0))
        bhk = col2.selectbox("🏠BHK: ",options=np.arange(1,6,1.0))
        bath = col2.selectbox("🛁 Bath Room: ",options=np.arange(1,6,1.0))

    #To get location from encoded location
    for i,j in zip(df['location'].unique(),df['encoded_loc'].unique()):
        if i==loc:
            location=j
            break
    data = [[location,sqft,bath,bhk]]

    #model
    with open('RFmodel.pkl','rb') as file:
        model = pickle.load(file)

    @st.dialog("🏡 House Details")
    def house_details(loc,sqft,bhk,bath,prediction):
        st.text(f"{'Location📍'.ljust(15)}: {loc}")
        st.text(f"{'Sqr.ft 📐'.ljust(20)}: {sqft}")
        st.text(f"{'BHK 🏠'.ljust(19)}: {bhk}")
        st.text(f"{'Bathrooms 🛁'.ljust(11)}: {bath}")
        st.subheader(f"Predicted Price: ₹ {np.round(prediction,2)}")
    c1,c2,c3 = st.columns(3)
    if c2.button('💰 Predict Price'):
        prediction = model.predict(data)[0]*100000
        with st.spinner('Predicting....'):
            time.sleep(1)
            house_details(loc,sqft,bhk,bath,prediction)

elif selection=="Dashboard":
    with st.container(border=True,height=300):
        with st.spinner("Loading..."):
            time.sleep(3)
            a1,a2 = st.columns(2)
            data = df.groupby("location")['price'].mean().reset_index().sort_values(by='price',ascending=False).iloc[:10,:]
            a1.bar_chart(x="location",y='price',data=data,y_label='Price (Lakhs)',color='location')

            bhk = df.groupby("bhk")['price'].mean().reset_index().sort_values(by='price',ascending=False)
            a2.bar_chart(x='bhk',y='price',data=df,y_label='price (lakhs)',color='bhk')

            a2.line_chart(y='price',x='total_sqft',data=df)

            fig = px.histogram(df,x='price',nbins=30,height=350)
            a1.plotly_chart(fig)
elif selection=='Sample Data':
    df = df.drop("Unnamed: 0",axis=1)
    st.dataframe(df.sample(20),hide_index=True,height=300)
