import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# Load your dataset
# Replace 'your_dataset.csv' with the path to your dataset file
path = "C:\\Users\\Ege Deniz\\Documents\\dataset.csv"
dtype_spec = {'albedo': float, 'H': float, 'diameter': float, 'e': float, 'a': float, 'q': float, 'i': float, 'n': float}
df = pd.read_csv(path, dtype=dtype_spec, low_memory=False)
df = df[['albedo', 'H', 'diameter', 'e', 'a', 'q', 'i', 'n']]


# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()