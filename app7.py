# //this app actually generates the string of lists and converts them into a data frame 
from flask import Flask, request, jsonify, render_template
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_huggingface import HuggingFaceEndpoint
import ast
metrics = []
# Initialize HuggingFaceEndpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
sec_key = "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"  # Replace with your Hugging Face token
llm = HuggingFaceEndpoint(repo_id=repo_id, max_lenlgth=1, temperature=1, token=sec_key)

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
        # self.metrics = []

    def update_context(self, key, value):
        self.context[key] = value

    def get_context(self, key):
        return self.context.get(key, None)

    def add_candidate_response(self, question, response):
        self.candidate_responses.append({'question': question, 'response': response})

    # def add_evaluation_metric(self, metric):
    #     self.evaluation_metrics.append(metric)

    def reset(self):
        self.context = {}
        self.candidate_responses = []
        self.evaluation_metrics = []

# Define chatbot
class InterviewBot:
    metrics = []
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

Accuracy (0-10): 
Clarity (0-10): 
Conciseness (0-10): 
Depth (0-10): 
Relevance (0-10):
generate a single line response of the metric values as a python list and dont give any explanation:
original answer: {original_answer}
user answer: {user_input}
"""
        print(llm.invoke(prompt))
        response = llm.invoke(prompt)
        #this will be a list where all the metrics are appended
        # try:
        metrics.append(list(response))
        # except Exception as e:
        #     print(f"Error evaluating response: {e}")
        #     metrics = metrics.append([0,0,0,0,0])
         
        # self.state.add_evaluation_metric(metrics)
        # print(metrics)
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
            self.is_job_position_verified = False
            self.state.reset()
            return "Interview terminated. Thank you for using the Interview Prep Chatbot."

        current_index = self.state.get_context("current_question_index") - 1
        questions = self.state.get_context("questions")
        if current_index < len(questions):
            current_question = questions[current_index]
            self.state.add_candidate_response(current_question, user_input)

            # Evaluate the answer
            apple = self.evaluate_answer(user_input, current_question)
            print(type(apple))
            print(f"cata metrics: {apple}")

            next_question = self.get_next_question()
            if next_question:
                return f"Next question: {next_question}"
            else:
                return self.complete_interview()
        else:
            return "No more questions."

    def complete_interview(self):
        df = pd.DataFrame(self.state.evaluation_metrics)
        print("Interview completed. Evaluation DataFrame:")
        print(df)
        self.state.reset()
        return self.create_dashboard(df)

    def create_dashboard(self, df):
        fig = go.Figure()

        for metric in df.columns:
            fig.add_trace(go.Bar(
                x=df.index,
                y=df[metric],
                name=metric
            ))

        fig.update_layout(
            title='Interview Performance Metrics',
            xaxis=dict(title='Question Number'),
            yaxis=dict(title='Score'),
            barmode='group'
        )

        fig_html = fig.to_html(full_html=False)
        return render_template('dashboard.html', fig_html=fig_html)

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

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
