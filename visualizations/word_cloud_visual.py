from data_preprocessing import DataProcessor
import matplotlib.pyplot as plt
from wordcloud import WordCloud

class WordCloudVisualizer:
    def __init__(self, text):
        """
        Initialize a WordCloudVisualizer object.

        Parameters:
        - text (str): Input text for generating a word cloud visualization.

        This class creates a word cloud visualization based on the input text, showing the frequency
        of the text's words within the trained model's dataset.
        """

        self.features = text.split()

        # Initialize the trained model's dataset
        data_processor = DataProcessor()
        df = data_processor.format_csv('data/inspiration.csv', ['ID', 'Image-link', 'Quote-url'])
        df = data_processor.process_data(df)

        # Initialize base frequencies
        self.frequencies = {word: 1 for word in self.features}

        # Count the frequency of each feature in the dataset
        for quote in df['quote']:
            quote_list = quote.split()
            for word in self.features:
                self.frequencies[word] += quote_list.count(word)

    def get_chart(self):
        """
        Get the generated word cloud chart.

        Returns:
        - word_cloud_chart (matplotlib.figure.Figure): A matplotlib figure object representing the word cloud chart.
        """
        
        plt.figure(figsize=(8, 3))
        if self.features:
            word_cloud = WordCloud(width=800, height=400, 
                                   background_color='white', contour_color='black',
                                   contour_width=3).generate_from_frequencies(self.frequencies)
            word_cloud.recolor(color_func=monochromatic_color_func)
            plt.imshow(word_cloud, interpolation='bilinear')
            plt.axis('off')

        return plt
    
# Set a monochromatic color function (all words will be the same color)
def monochromatic_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return 'green'
