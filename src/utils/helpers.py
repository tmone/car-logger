def normalize(data):
    """Normalize the given dataset."""
    if not data:
        return data
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

def filter_data(data, criteria):
    """Filter the dataset based on specified criteria."""
    return [item for item in data if criteria(item)]