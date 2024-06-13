import pandas as pd
import numpy as np
import pickle
import streamlit as st
import plotly.graph_objects as go

def get_clean_data():
    data = pd.read_csv("Advanced_IoT_Dataset.csv")
    data.Random = data.Random.map({"R1": 1, "R2": 2, "R3": 3})
    data.Class = data.Class.map({"SA": 1, "SB": 2, "SC": 3, "TA": 4, "TB": 5, "TC": 6})
    data = data.rename(columns={
        " Average  of chlorophyll in the plant (ACHP)": "ACPH",
        " Plant height rate (PHR)": "PHR",
        "Average wet weight of the growth vegetative (AWWGV)": "AWWGV",
        "Average leaf area of the plant (ALAP)": "ALAP",
        "Average number of plant leaves (ANPL)": "ANPL",
        "Average root diameter (ARD)": "ARD",
        " Average dry weight of the root (ADWR)": "ADWR",
        " Percentage of dry matter for vegetative growth (PDMVG)": "PDMVG",
        "Average root length (ARL)": "ARL",
        "Average wet weight of the root (AWWR)": "AWWR",
        " Average dry weight of vegetative plants (ADWV)": "ADWV",
        "Percentage of dry matter for root growth (PDMRG)": "PDMRG"
    })
    return data

def add_sidebar():
    st.sidebar.header("Inputs")

    data = get_clean_data()

    slider_label = [
        ("Random", "Random"), ("ACPH", "ACPH"), ("PHR", "PHR"), ("AWWGV", "AWWGV"),
        ("ALAP", "ALAP"), ("ANPL", "ANPL"), ("ARD", "ARD"), ("ADWR", "ADWR"),
        ("PDMVG", "PDMVG"), ("ARL", "ARL"), ("AWWR", "AWWR"), ("ADWV", "ADWV"),
        ("PDMRG", "PDMRG")
    ]

    input_dict = {}

    for key, label in slider_label:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max())
        )

    return input_dict


def get_renderers_Framework(input_dict):
    data_values = list(map(float, input_dict.values()))
    fig = go.Figure(data=[go.Bar(x=data_values)],
                    layout_title_text="A Figure Displayed with fig.show()")
    fig = go.Figure()   
    return fig


def get_scaled_values(input_dict, scaler):
    input_array = np.array(list(input_dict.values())).reshape(1, -1)
    scaled_array = scaler.transform(input_array)
    return scaled_array

def add_prediction(input_data):
    with open("model.pkl", "rb") as model_in:
        model = pickle.load(model_in)
    with open("scaler.pkl", "rb") as scaler_in:
        scaler = pickle.load(scaler_in)

    input_scaled = get_scaled_values(input_data, scaler)
    prediction = model.predict(input_scaled)

    probabilities = model.predict_proba(input_scaled)[0]
    class_labels = ["SA", "SB", "SC", "TA", "TB", "TC"]
    predicted_class = class_labels[prediction[0] - 1]
    
    st.write(f"The predicted class is: {predicted_class}")


def main():
    st.set_page_config(page_title="Agriculture Advanced IoT Prediction", page_icon=":seedling:", layout="wide", initial_sidebar_state="expanded")

    st.title("Agriculture Advanced IoT Prediction")
    st.write("This ML app is used to predict the class of the IoT based on the inputs provided")

    input_data = add_sidebar()

    col1,col2 = st.columns([4,1])

    with col1:
        renderers = get_renderers_Framework(input_data)
        st.plotly_chart(renderers)
    with col2:
        if st.button("Predict"):
            add_prediction(input_data)
        if st.button("About"):
            st.write("This app is built with Streamlit")
            st.write("It helps in predicting IoT class based on user inputs")

if __name__ == "__main__":
    main()
