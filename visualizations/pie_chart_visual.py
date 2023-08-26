import matplotlib.pyplot as plt
import numpy as np

class PieChart:
    def __init__(self, top_categories, category_percentages):
        total_percentages = np.sum(category_percentages)
        normalized_percentages = [percentage / total_percentages * 100.0 for percentage in category_percentages]

        fig, ax = plt.subplots()
        ax.pie(normalized_percentages, labels=top_categories, autopct='%1.1f%%', startangle=90)

        self.pie_chart = fig
    
    def get_chart(self):
        return self.pie_chart
