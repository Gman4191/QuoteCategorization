# Inspirational Quote Categorization

This project is a machine learning application that categorizes inspirational quotes into different categories. It uses a trained Complement Naive Bayes classifier to analyze input quotes and predict the most relevant categories.

### Hosted Application

You can access the live version of this application [here](https://quotecategorization.streamlit.app/).

Feel free to try it out and explore the features!


### Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/Gman4191/QuoteCategorization.git

2. Navigate to the project directory:
   
   ```bash
   cd QuoteCategorization

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   
3. Run the application:
   
   ```bash
   streamlit run main.py 

4. Enter a quote in the input field to see the categorization results.

### Example Usage

Let's take a simple example input for the application:

Input Quote: "It’s hard to beat a person who never gives up."

Expected Categories:
- "Hard Work"
- "Perseverance"

#### Feature Importance Chart

The feature importance chart shows the importance of each word in the quote for category prediction. In this example, the words "hard," "gives," and "person" significantly contribute to the chosen categories. On the other hand, "beat" is also shown in the chart, but its inclusion does not affect the categorization outcome.

#### Word Frequency Visualization

The word frequency visualization provides insights into how often each of the important words is found in the dataset used to train the model. It helps users understand the prevalence of specific words within the context of the inspirational quotes in the dataset.

This example demonstrates how the application categorizes an input quote and provides additional information about the significance of individual words in the prediction process.
