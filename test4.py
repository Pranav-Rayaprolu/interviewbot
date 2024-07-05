# //this will integrate metric metric_values
from flask import Flask, request, jsonify, render_template
import pandas as pd

app = Flask(__name__)

# Assume df is your DataFrame with evaluation metrics
df = pd.DataFrame({
    'Accuracy': [0, 6, 9],
    'Clarity': [0, 7, 7],
    'Conciseness': [0, 8, 7],
    'Depth': [0, 3, 5],
    'Relevance': [0, 7, 10]
})

@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify(df.to_dict(orient='records'))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
