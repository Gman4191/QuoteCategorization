import streamlit as st
import pickle
from data_preprocessing import DataProcessor
from visualizations.pie_chart_visual import PieChart
from visualizations.feature_importance_visual import FeatureImportanceChart
from visualizations.word_cloud_visual import WordCloudVisualizer

# Constants and configurations
MODEL_FILE_PATH = 'models/trained_model.pkl'
TOP_N_CATEGORIES = 5

# Initialize Streamlit page
st.set_page_config(layout="wide")
container = st.container()
container.header("Inspirational Quote Categorization")
quote = container.text_input("Enter a quote")

def load_model(file_path):
    """
    Load a pretrained model from the given file path.
    """
    with open(file_path, 'rb') as model_file:
        return pickle.load(model_file)

def main():
    if quote:
        # Load the pretrained model
        loaded_model = load_model(MODEL_FILE_PATH)

        # Preprocess the new text input
        data_processor = DataProcessor()
        processed_quote = data_processor.process_text(quote)

        # Make predictions using the loaded model
        predicted_probabilities = loaded_model.predict_proba([processed_quote])

        # Get the top N categories
        top_indices = predicted_probabilities.argsort()[0][-TOP_N_CATEGORIES:][::-1]
        top_categories = loaded_model.classes_[top_indices]
        top_probabilities = predicted_probabilities[0][top_indices]

        display_results(top_indices, top_categories, top_probabilities, processed_quote, loaded_model)

def display_results(top_indices, top_categories, top_probabilities, processed_quote, loaded_model):
    """
    Display the top categories, feature importance chart, and word frequency visualization.
    """

    # Display the top category pie chart
    pie_chart = PieChart(top_categories, top_probabilities)
    col_1, col_2 = st.columns(2)
    col_1.subheader("Top 5 Categories")
    col_1.pyplot(pie_chart.get_chart())

    # Display the feature importance chart
    feature_importance_chart = FeatureImportanceChart(features=processed_quote, top_category_index=top_indices[0], model=loaded_model)
    col_2.subheader("Feature Importance Chart")
    col_2.pyplot(feature_importance_chart.get_chart())

    # Display the word frequency visualization
    word_cloud = WordCloudVisualizer(processed_quote)
    st.subheader("Word Frequency within Training Data Set")
    st.pyplot(word_cloud.get_chart())

if __name__ == "__main__":
    main()
