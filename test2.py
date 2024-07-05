import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have your data and plotting logic here

# Example plot generation (modify as per your actual plot code)
data = [
    [0, 10, 10, 0, 10],
    [5, 6, 8, 3, 9],
    [6, 5, 6, 3, 8]
]
columns = ["Accuracy", "Clarity", "Conciseness", "Depth", "Relevance"]
df = pd.DataFrame(data, columns=columns)

# Generate a bar chart
plt.figure(figsize=(8, 6))
for column in columns:
    plt.plot(df[column], marker='o', label=column)
plt.xlabel('Interview Rounds')
plt.ylabel('Score')
plt.title('Interview Evaluation Scores')
plt.legend()
plt.grid(True)

# Save the plot as an image
plt.savefig('static/overview_plot.png')  # Adjust path as per your project structure
plt.close()
