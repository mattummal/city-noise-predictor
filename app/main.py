import streamlit as st

st.set_page_config(page_title='CityNoisePredictor', page_icon='☂️')

from weather import weather_content
from prediction_noise import prediction_noise_content
from historical_noise import historical_noise_content

# Create a dictionary to map page names to functions
pages = {
	'Quality of Weather & Air': weather_content,
	'Predicting the Noise': prediction_noise_content,
	'Noise Data History': historical_noise_content,
}


# Add a sidebar to select the page
selected_page = st.sidebar.selectbox('Select a page', list(pages.keys()))


# Display the selected page
pages[selected_page]()


# About
st.sidebar.markdown("<h2 style='text-align: left;'>About</h2>", unsafe_allow_html=True)
st.sidebar.markdown(
	"<p style='text-align: left;'>Welcome to City Noise Predictor, a Web App developed by for the final project. <br>\
        Our app predicts noise levels in New York city using weather and air quality data. \
        With accurate forecasts, you can plan activities, minimize disruptions, and maintain a peaceful environment. <br>\
            </p>",
	unsafe_allow_html=True,
)

st.sidebar.info(
	'**Source code: [@repo](https://github.com/mattummal/city-noise-predictor.git)**', icon='💡'
)
