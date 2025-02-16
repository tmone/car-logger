class DataAnalyzer:
    def __init__(self):
        self.results = None

    def analyze(self, data):
        # Perform data analysis here
        # Example: Calculate mean, median, etc.
        self.results = {
            'mean': data.mean(),
            'median': data.median(),
            'std_dev': data.std()
        }

    def get_results(self):
        return self.results