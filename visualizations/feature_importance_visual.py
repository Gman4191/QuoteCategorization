import matplotlib.pyplot as plt
import numpy as np

class FeatureImportanceChart:
    def __init__(self, features, top_category_index, model):
        self.features = features.split()
        self.importances = []

        baseline_probability = model.predict_proba([features])[0, top_category_index]

        for feature in self.features:
            # All the features minus the current feature
            new_features = [f for f in self.features if f != feature]
            modified_probability = model.predict_proba([' '.join(new_features)])[0, top_category_index]
            self.importances.append(np.abs(baseline_probability - modified_probability))
        
        total_importance = np.sum(self.importances)
        if total_importance > 0.0:
            self.importances = [importance / total_importance * 100.0 for importance in self.importances]

    def get_chart(self):
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