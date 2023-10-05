import matplotlib.pyplot as plt
import numpy as np

class PieChart:
    def __init__(self, top_categories, category_percentages):
        """
        Initialize a PieChart object.

        Parameters:
        - top_categories (list): A list of category labels.
        - category_percentages (list): A list of percentage values corresponding to each category.

        This class creates a pie chart visualization to represent the distribution of categories
        and their respective percentages.
        """

        total_percentages = np.sum(category_percentages)
        normalized_percentages = [percentage / total_percentages * 100.0 for percentage in category_percentages]

        fig, ax = plt.subplots()
        ax.pie(normalized_percentages, labels=top_categories, autopct='%1.1f%%', startangle=90)

        self.pie_chart = fig
    
    def get_chart(self):
        """
        Get the generated pie chart.

        Returns:
        - pie_chart (matplotlib.figure.Figure): A matplotlib figure object representing the pie chart.
        """
        return self.pie_chart
