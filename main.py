from visualizations.pie_chart_visual import PieChart
from visualizations.feature_importance_visual import FeatureImportanceChart
from visualizations.word_cloud_visual import WordCloudVisualizer
from data_preprocessing import DataProcessor
import streamlit as st
import pickle

st.set_page_config(layout="wide")

container = st.container()
container.header("Inspirational Quote Categorization")
quote = container.text_input("Enter a quote")

# Load the pretrained model
with open('models/trained_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Preprocess the new text input
data_processor = DataProcessor()
quote = data_processor.process_text(quote)

if quote:
    # Make predictions using the loaded model
    predicted_probabilities = loaded_model.predict_proba([quote])

    # Get the top N categories
    top_n = 5
    top_indices = predicted_probabilities.argsort()[0][-top_n:][::-1]
    top_categories = loaded_model.classes_[top_indices]
    top_probabilities = predicted_probabilities[0][top_indices]

    pie_chart = PieChart(top_categories, top_probabilities)

    col_1, col_2 = st.columns(2)
    col_1.subheader("Top 5 Categories")
    col_1.pyplot(pie_chart.get_chart())

    feature_importance_chart = FeatureImportanceChart(features=quote, top_category_index=top_indices[0], model=loaded_model)
    col_2.subheader("Feature Importance Chart")
    col_2.pyplot(feature_importance_chart.get_chart())

    word_cloud = WordCloudVisualizer(quote)
    st.subheader("Word Frequency within Training Data Set")
    st.pyplot(word_cloud.get_chart())