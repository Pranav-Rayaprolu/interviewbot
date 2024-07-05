import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

def create_visualizations(df):
    columns = df.columns
    
    # Example: Plotting mean scores
    mean_scores = df.mean()
    plt.figure(figsize=(8, 6))
    plt.bar(columns, mean_scores, color='#1f77b4')
    plt.xlabel('Evaluation Criteria')
    plt.ylabel('Mean Score')
    plt.title('Mean Evaluation Scores')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save the plot as an image
    plt.savefig('static/mean_scores.png')  # Adjust path as per your project structure
    plt.close()
    
    # Example: Plotting pie chart
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    explode = (0.1, 0, 0, 0, 0)
    plt.pie(mean_scores, labels=mean_scores.index, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode)
    plt.axis('equal')
    plt.title('Mean Evaluation Scores Distribution')
    
    # Save the pie chart as an image
    plt.savefig('static/mean_scores_pie.png')  # Adjust path as per your project structure
    plt.close()
# Assuming df is your DataFrame
data = [
    [0, 10, 10, 0, 10],
    [5, 6, 8, 3, 9],
    [6, 5, 6, 3, 8]
]
columns = ["Accuracy", "Clarity", "Conciseness", "Depth", "Relevance"]
df = pd.DataFrame(data, columns=columns)

# Call the visualization function with your DataFrame
create_visualizations(df)