from flask import Flask, request, jsonify, render_template
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import os
matplotlib.use('Agg')

# Initialize HuggingFaceEndpoint (Dummy for illustration)
class HuggingFaceEndpoint:
    def __init__(self, repo_id, max_length, temperature, token):
        pass

    def invoke(self, prompt):
        return str({
            "Accuracy": 8,
            "Clarity": 7,
            "Conciseness": 6,
            "Depth": 5,
            "Relevance": 9
        })

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
sec_key = "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"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=1, token=sec_key)

# Load dataset
def load_dataset(filepath):
    df = pd.read_csv(filepath)
    print("Dataset columns:", df.columns)  # Debugging output to check column names
    return df

# Define conversation state
class ConversationState:
    def __init__(self):
        self.context = {}
        self.candidate_responses = []
        self.evaluation_metrics = []

    def update_context(self, key, value):
        self.context[key] = value

    def get_context(self, key):
        return self.context.get(key, None)

    def add_candidate_response(self, question, response):
        self.candidate_responses.append({'question': question, 'response': response})

    def add_evaluation_metric(self, metric):
        self.evaluation_metrics.append(metric)

    def reset(self):
        self.context = {}
        self.candidate_responses = []
        self.evaluation_metrics = []

# Define chatbot
class InterviewBot:
    def __init__(self, dataset):
        self.state = ConversationState()
        self.dataset = dataset
        self.is_job_position_verified = False

    def get_questions_for_position(self, company, job_position):
        return self.dataset[(self.dataset['company_name'].str.lower() == company.lower()) & 
                            (self.dataset['Job_Position'].str.lower() == job_position.lower())]['Questions'].tolist()

    def get_answer_for_question(self, question):
        return self.dataset[self.dataset['Questions'] == question]['Answers'].values[0]

    def start_interview(self, company, job_position):
        self.state.update_context("company", company)
        self.state.update_context("job_position", job_position)
        self.state.update_context("questions", self.get_questions_for_position(company, job_position))
        self.state.update_context("current_question_index", 0)
        return self.get_next_question()

    def get_next_question(self):
        questions = self.state.get_context("questions")
        current_index = self.state.get_context("current_question_index")
        if current_index < len(questions):
            question = questions[current_index]
            self.state.update_context("current_question_index", current_index + 1)
            return question
        else:
            return None

    def verify_job_position(self, company, job_position):
        available_positions = self.dataset[['company_name', 'Job_Position']].apply(lambda x: (x['company_name'].lower(), x['Job_Position'].lower()), axis=1).tolist()
        if (company.lower(), job_position.lower()) in available_positions:
            first_question = self.start_interview(company, job_position)
            return True, first_question
        else:
            return False, None

    def evaluate_answer(self, user_input, question):
        original_answer = self.get_answer_for_question(question)
        prompt = f"""
You are an AI model tasked with evaluating the quality of answers based on several key metrics. These metrics are used to determine the effectiveness and clarity of the responses. You will be given two inputs: the original answer and the user's answer. Your task is to compare the user's answer with the original answer and generate an array of metrics for grading the user's answer.

The metrics are as follows:

Accuracy (0-10): How accurately the user's answer reflects the definition and differences between parametric and non-parametric models.
Clarity (0-10): How clear and understandable the user's answer is for someone who may not be familiar with the topic.
Conciseness (0-10): How concise and to the point the user's answer is without unnecessary information.
Depth (0-10): How well the user's answer explains the concepts, including any relevant details.
Relevance (0-10): How relevant the user's answer is to the question asked.
generate a one line response of the metric values as a python dictionary data structure:
original answer: {original_answer}
user answer: {user_input}
"""
        response = llm.invoke(prompt)
        try:
            metrics = eval(response)
            if not isinstance(metrics, dict):
                raise ValueError("Response is not a valid dictionary")
        except Exception as e:
            print(f"Error evaluating response: {e}")
            metrics = {
                "Accuracy": 0,
                "Clarity": 0,
                "Conciseness": 0,
                "Depth": 0,
                "Relevance": 0
            }
        self.state.add_evaluation_metric(metrics)
        return metrics

    def respond(self, user_input, company=None, domain=None):
        if company and domain and not self.is_job_position_verified:
            is_verified, first_question = self.verify_job_position(company, domain)
            if is_verified:
                self.is_job_position_verified = True
                return f"Got it. Starting interview for {self.state.get_context('job_position')} at {self.state.get_context('company')}. Here is your first question: {first_question}"
            else:
                return "Please provide a valid company name and job position in the format 'Company, Job Position'."

        if user_input.lower() == "quit":
            self.complete_interview()
            self.is_job_position_verified = False
            self.state.reset()
            return "Interview terminated. Thank you for using the Interview Prep Chatbot."

        current_index = self.state.get_context("current_question_index") - 1
        questions = self.state.get_context("questions")
        if current_index < len(questions):
            current_question = questions[current_index]
            self.state.add_candidate_response(current_question, user_input)

            # Evaluate the answer
            metrics = self.evaluate_answer(user_input, current_question)
            print(f"Evaluation metrics: {metrics}")

            next_question = self.get_next_question()
            if next_question:
                return f"Next question: {next_question}"
            else:
                return self.complete_interview()
        else:
             print("No more questions.")
             return self.complete_interview()

    def complete_interview(self):
        df = pd.DataFrame(self.state.evaluation_metrics)
        print("Interview completed. Evaluation DataFrame:")
        print(df)
        self.state.reset()
        # self.create_visualizations(df)
        return df

    def create_visualizations(self, df):
        # Example: Create visualizations
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
        
        # Save the plot as an image in a static directory
        if not os.path.exists('static'):
            os.makedirs('static')
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

app = Flask(__name__)
dataset = load_dataset('all_companies2.csv')
bot = InterviewBot(dataset)

@app.route('/')
def index():
    return render_template('pageA.html')

@app.route('/pageB', methods=['POST'])
def pageB():
    company = request.form.get('company')
    return render_template('pageB.html', company=company)

@app.route('/pageC', methods=['POST'])
def pageC():
    company = request.form.get('company')
    domain = request.form.get('domain')
    return render_template('pageC.html', company=company, domain=domain)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    company = request.json.get('company')
    domain = request.json.get('domain')
    response = bot.respond(user_input, company=company, domain=domain)
    return jsonify({'response': response})

@app.route('/get_df',methods=['GET'])
def get_df():
    df = bot.complete_interview()
    return df.to_json(orient='records')

@app.route('/dashboard')
def dashboard():
    df = bot.complete_interview()
    metrics = df.mean().to_dict()
    return render_template('dashboard1.html', metrics=metrics)

@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    df = bot.complete_interview()
    print(df)
    # metrics = df.mean().to_dict()
    # return jsonify(metrics)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
