import matplotlib.pyplot as plt
import pandas as pd

# Create the data
data = [
    [0, 10, 10, 0, 10],
    [5, 4, 7, 3, 8],
    [0, 0, 0, 0, 0]
]

# Create column names
columns = ["Accuracy", "Clarity", "Conciseness", "Depth", "Relevance"]

# Create the DataFrame
df = pd.DataFrame(data, columns=columns)

# Calculate mean scores for each metric
mean_scores = df.mean()

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

# Plotting the bar chart on the first subplot (ax1)
index = df.index
metric_values = df.values

num_columns = df.shape[1]
width = 0.35
bar_positions = [i + width/2 for i in range(num_columns)]

# Using shades of blue for the bar chart
bar_colors = ['#1f77b4', '#aec7e8', '#6baed6', '#3182bd', '#08519c']

for i in range(num_columns):
    ax1.bar(bar_positions[i], metric_values[:, i], width, label=columns[i], color=bar_colors[i])

ax1.set_xlabel("Evaluation Criteria")
ax1.set_ylabel("Score")
ax1.set_title("Interview Evaluation Scores")
ax1.set_xticks(bar_positions)
ax1.set_xticklabels(columns)
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.6)

# Plotting the pie chart on the second subplot (ax2)
# Using shades of blue for the pie chart
pie_colors = ['#1f77b4', '#aec7e8', '#6baed6', '#3182bd', '#08519c']
explode = (0.1, 0, 0, 0, 0)

ax2.pie(mean_scores, labels=mean_scores.index, autopct='%1.1f%%', startangle=140, colors=pie_colors, explode=explode)
ax2.set_title('Mean Evaluation Scores')
ax2.axis('equal')

# Adjust layout and display the plot
plt.savefig('static/overview_plot.png')  # Adjust path as per your project structure
plt.close()