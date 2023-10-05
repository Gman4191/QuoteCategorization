import matplotlib.pyplot as plt
import numpy as np

class FeatureImportanceChart:
    def __init__(self, features, top_category_index, model):
        """
        Initialize a FeatureImportanceChart object.

        Parameters:
        - features (str): A string containing space-separated features.
        - top_category_index (int): The index of the top category of interest.
        - model: A machine learning model used for prediction.

        This class calculates and visualizes the importance of individual features in predicting
        the top category, by comparing prediction probabilities when each feature is removed.
        """

        # Get a list of individual features from the input features text
        self.features = features.split()
        self.importances = []

        # Calculate the baseline prediction probability for the top category
        baseline_probability = model.predict_proba([features])[0, top_category_index]

        for feature in self.features:
            # All the features minus the current feature
            new_features = [f for f in self.features if f != feature]

            # Calculate the modified prediction probability
            modified_probability = model.predict_proba([' '.join(new_features)])[0, top_category_index]

            # Calculate and store the importance as the absolute difference
            self.importances.append(np.abs(baseline_probability - modified_probability))
        
        total_importance = np.sum(self.importances)
        if total_importance > 0.0:
            self.importances = [importance / total_importance * 100.0 for importance in self.importances]

    def get_chart(self):
        """
        Generate and return a feature importance bar chart.

        Returns:
        - chart (matplotlib.figure.Figure): A matplotlib figure object representing the chart.
        """
        plt.figure(figsize=(10, 6))
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.ylim(0, 100)
        plt.title('Feature Importance')
        plt.tight_layout()

        if self.features:
            plt.bar(self.features, self.importances)
        self.chart = plt
        return self.chart