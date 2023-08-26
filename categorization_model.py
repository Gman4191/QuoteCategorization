from data_preprocessing import DataProcessor
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

class QuoteClassificationModel:
    def __init__(self) -> None:
        self.k = 100

    def train_and_evaluate(self, processed_df):
        pipeline = Pipeline([
            ['vectorizer', TfidfVectorizer()],
            ['select_k_best', SelectKBest()],
            ['scaler', MaxAbsScaler()],
            ['clf', ComplementNB(force_alpha=True)]
        ])

        # Define the hyperparameter grid
        param_grid = {
            'vectorizer__min_df': [2, 3, 4],
            'vectorizer__max_df': [.7, .8, .9],
            'select_k_best__k': [500, 600, 700],
            'select_k_best__score_func' : [chi2],
        }

        features, labels = processed_df['quote'], processed_df['category']

        # Split the data into training and testing sets
        features_train, features_test, labels_train, labels_test = train_test_split(
            features, labels, test_size=0.2, random_state=40
        )
        # Initialize GridSearchCV
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=-1)

        # Fit the grid search on the training data
        grid_search.fit(features_train, labels_train)

        # Get the best estimator from the grid search
        best_estimator = grid_search.best_estimator_

        # Predict using the best model
        labels_prediction = best_estimator.predict(features_test)

        # Get the unique class labels
        unique_labels = processed_df['category'].unique()

        # Calculate accuracy
        accuracy = accuracy_score(labels_test, labels_prediction)
        print("Best Model Accuracy:", accuracy)

        # Calculate precision, recall, and F1-score
        precision = precision_score(labels_test, labels_prediction, average='weighted', zero_division=1)
        recall = recall_score(labels_test, labels_prediction, average='weighted', zero_division=1)
        f1 = f1_score(labels_test, labels_prediction, average='weighted')

        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1)

        # Create a confusion matrix
        # conf_matrix = confusion_matrix(labels_test, labels_prediction, labels=unique_labels)
        # plt.figure(figsize=(12, 8))
        # sns.heatmap(conf_matrix, annot=False, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
        # plt.xlabel("Predicted")
        # plt.ylabel("Actual")
        # plt.title("Confusion Matrix")
        # plt.show()
        return grid_search.best_estimator_

# Example
data_processor = DataProcessor()
df = data_processor.format_csv('data/inspiration.csv')
model = QuoteClassificationModel()
processed_df = data_processor.process_data(df)
best_model = model.train_and_evaluate(processed_df)
print(best_model)

with open('models/trained_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)
