#House price predictor streamlit app
import json
import time
import pickle
import pandas as pd
import streamlit as st
import plotly.express as px #install this package
from streamlit_lottie import st_lottie #install this package

#load the model
with open("RF_Model.pkl",'rb') as file:
    model = pickle.load(file)

st.set_page_config(page_title="Bengaluru House price predictor",page_icon="icons/house_logo.png",initial_sidebar_state="expanded",layout="wide")
st.sidebar.subheader("House Price Predictor")
st.sidebar.image("icons/house_logo.png",width=150)

st.subheader("Welcome! to House Price Predictor 💰")
#load the data
df = pd.read_csv("cleaned.csv")
df = df.drop("Unnamed: 0",axis=1)
#loading anime
def load_anime():
    with open("home_anime.json",'rb') as f:
        anime = json.load(f)
        return anime

#dialog box

@st.dialog("🏠 Your Details")
def display(loc,sqft,bath,bhk,prediction):
    st.markdown(f'''
                📍**Location**: {loc}

                📐 **Total sqft**: {sqft}

                🛁 **Bath**: {bath}

                🏠 **Bhk**: {bhk}
                
                ''')
    st.subheader(f"Predicted Price: ₹ {round(prediction*100000)}")
selection = st.segmented_control(None,options=['Home',"Predict Price","Dashboard","Sample Data"],default="Home")

if selection == 'Home':
    col1,col2 = st.columns([1,2])
    with col1:
        anime = load_anime()
        st_lottie(anime,width=250)

    with col2.container(border=True,height=300):
        st.markdown("""🏠 About This App
                    
Welcome to the House Price Prediction App!
This tool helps you estimate the price of a house based on key features like location, square footage, number of bedrooms, and number of bathrooms.

📍 Dataset Used:
                    
The predictions and charts are based on real data from the Bangalore House Price dataset, which contains detailed information about residential properties in Bangalore.

✨ What You Can Do:
                    
📊 Dashboard: 
                    Explore charts and visualizations to understand Bangalore’s house price trends.

💰 Price Prediction: 
                    Get an estimated price for your property based on your inputs.
🗂️ Sample Data:       
             View a sample of the Bangalore housing dataset.
This app is built with Streamlit and powered by a Machine Learning model trained on Bangalore housing market data. Use it for learning, exploration, or quick insights — but always consult real market experts for final decisions!

Happy predicting! 🏡✨
        """)
elif selection == "Predict Price":
    with st.container(border=True):
        c1,c2 = st.columns(2)
        loc = c1.selectbox("📍Location: ",options=df['location'].unique())
        str_loc = loc
        sqft = c1.number_input("📐 Total sq.ft",min_value=300.0,max_value=40000.0,step=1000.0)
        bhk = c2.selectbox("🏡 BHK: ",options=[1.0,2.0,3.0,4.0,5.0])
        bath = c2.selectbox("🛁 Bath: ",options=[1.0,2.0,3.0,4.0,5.0])

        for i,j in zip(df['location'],df['encoded_loc']):
            if i==loc:
                loc=j
                break
        
        data = [[loc,sqft,bath,bhk]]

    a1,a2,a3 = st.columns([1.4,1,1])
    if a2.button("Predict Price"):
        prediction = model.predict(data)
        display(str_loc,sqft,bath,bhk,prediction[0])

elif selection == "Dashboard":
    with st.container(border=True,height=300):
        with st.spinner("Loading..."):
            time.sleep(2)
            c1,c2 = st.columns(2)
            loc_price = df.groupby('location')['price'].mean().reset_index().sort_values(by='price',ascending=False)
            c1.bar_chart(x='location',y='price',data=loc_price.iloc[:7,:],color='location')
            c2.bar_chart(x='bhk',y='price',data=df,color='bhk')

            b1,b2 = st.columns(2)
            fig  = px.histogram(df,x='price',height=350)
            b1.plotly_chart(fig)

            b2.line_chart(x='total_sqft',y='price',data=df,height=350)
elif selection == "Sample Data":
    st.dataframe(df.sample(20),height=300,hide_index=True)