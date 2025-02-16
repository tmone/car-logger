class DataPlotter:
    def plot(self, data):
        import matplotlib.pyplot as plt
        
        # Example plot (customize as needed)
        plt.figure(figsize=(10, 6))
        plt.plot(data['x'], data['y'], marker='o')
        plt.title('Data Visualization')
        plt.xlabel('X-axis Label')
        plt.ylabel('Y-axis Label')
        plt.grid()
        plt.show()

    def save_plot(self, filename):
        import matplotlib.pyplot as plt
        
        # Save the current figure
        plt.savefig(filename)