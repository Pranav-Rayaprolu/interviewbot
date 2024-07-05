from flask import Flask, request, jsonify, render_template
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import os
from langchain_huggingface import HuggingFaceEndpoint

matplotlib.use('Agg')

metrics = {
    "Accuracy": 0,
    "Clarity": 0,
    "Conciseness": 0,
    "Depth": 0,
    "Relevance": 0,
    "Total_no_of_Questions": 0
}

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
sec_key = "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"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_lenlgth=128, temperature=1, token=sec_key)

def load_dataset(filepath):
    df = pd.read_csv(filepath)
    return df

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

            metrics = self.evaluate_answer(user_input, current_question)
            next_question = self.get_next_question()
            if next_question:
                return f"Next question: {next_question}"
            else:
                return self.complete_interview()
        else:
            return self.complete_interview()

    def complete_interview(self):
        df = pd.DataFrame(self.state.evaluation_metrics)
        self.state.reset()
        self.create_visualizations(df)
        total = len(df)
        metrics['Accuracy'] = round((df['Accuracy'].sum() / total) * 10, 2)
        metrics['Clarity'] = round((df['Clarity'].sum() / total) * 10, 2)
        metrics['Conciseness'] = round((df['Conciseness'].sum() / total) * 10, 2)
        metrics['Depth'] = round((df['Depth'].sum() / total) * 10, 2)
        metrics['Relevance'] = round((df['Relevance'].sum() / total) * 10, 2)
        metrics['Total_no_of_Questions'] = len(df)

    def create_visualizations(self, df):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        columns = df.columns
        bar_colors = ['#1f77b4', '#aec7e8', '#6baed6', '#3182bd', '#08519c']
        mean_scores = df.mean()
        ax1.bar(columns, mean_scores, color=bar_colors)
        ax1.set_xlabel('Evaluation Criteria')
        ax1.set_ylabel('Mean Score')
        ax1.set_title('Mean Evaluation Scores')
        ax1.grid(axis='y', linestyle='--', alpha=0.6)
        ax2.pie(mean_scores, labels=mean_scores.index, autopct='%1.1f%%', startangle=140, colors=bar_colors, explode=(0.1, 0, 0, 0, 0))
        ax2.axis('equal')
        ax2.set_title('Mean Evaluation Scores Distribution')
        plt.savefig('static/overview_plot.png')
        plt.close()

app = Flask(__name__)
dataset = load_dataset('all_companies2.csv')
bot = InterviewBot(dataset)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/skill', methods=['POST'])
def skill():
    company = request.form.get('company')
    return render_template('skill.html', company=company)

@app.route('/chat', methods=['POST'])
def chat():
    company = request.form.get('company')
    domain = request.form.get('domain')
    return render_template('chat.html', company=company, domain=domain)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('message')
    company = request.json.get('company')
    domain = request.json.get('domain')
    response = bot.respond(user_input, company=company, domain=domain)
    return jsonify({'response': response})

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    return render_template('dashboard.html', metrics=metrics)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
