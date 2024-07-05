from flask import Flask, request, jsonify, render_template
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import ast

# Initialize HuggingFaceEndpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
sec_key = "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"  # Replace with your Hugging Face token
llm = HuggingFaceEndpoint(repo_id=repo_id, max_lenlgth=128, temperature=1, token=sec_key)

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
        self.memory = ConversationBufferMemory()
        
        # Define a prompt template
        self.prompt_template = PromptTemplate(template="""
You are an AI model tasked with evaluating the quality of answers based on several key metrics. These metrics are used to determine the effectiveness and clarity of the responses. You will be given pairs of original answers and the user's answers. Your task is to compare each user's answer with the original answer and generate an array of metrics for grading the user's answer.

The metrics are as follows:

Accuracy (0-10): How accurately the user's answer reflects the definition and differences between parametric and non-parametric models.
Clarity (0-10): How clear and understandable the user's answer is for someone who may not be familiar with the topic.
Conciseness (0-10): How concise and to the point the user's answer is without unnecessary information.
Depth (0-10): How well the user's answer explains the concepts, including any relevant details.
Relevance (0-10): How relevant the user's answer is to the question asked.

For each pair of answers, generate a one-line response of the metric values as a Python dictionary data structure.
{all_answers}
""")
        self.llm_chain = LLMChain(llm=llm, prompt=self.prompt_template, memory=self.memory)

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
        self.state.update_context("user_answers", [])
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

    def collect_user_answer(self, user_input):
        user_answers = self.state.get_context("user_answers")
        current_index = self.state.get_context("current_question_index") - 1
        questions = self.state.get_context("questions")
        if current_index < len(questions):
            current_question = questions[current_index]
            user_answers.append((current_question, user_input))
            self.state.update_context("user_answers", user_answers)

    def evaluate_all_answers(self):
        user_answers = self.state.get_context("user_answers")
        all_answers = "\n".join([f"Original answer: {self.get_answer_for_question(q)}\nUser answer: {a}" for q, a in user_answers])
        response = self.llm_chain.run(all_answers=all_answers)
        
        metrics_list = []
        try:
            metrics_list = [eval(m) for m in response.strip().split('\n')]
        except Exception as e:
            print(f"Error evaluating responses: {e}")
            metrics_list = [{"Accuracy": 0, "Clarity": 0, "Conciseness": 0, "Depth": 0, "Relevance": 0} for _ in user_answers]
        
        df = pd.DataFrame(metrics_list)
        df['Question Number'] = range(1, len(df) + 1)
        print(df) # this is to see the df
        return df

    def respond(self, user_input, company=None, domain=None):
        if company and domain and not self.is_job_position_verified:
            is_verified, first_question = self.verify_job_position(company, domain)
            if is_verified:
                self.is_job_position_verified = True
                return jsonify({'response': f"Got it. Starting interview for {self.state.get_context('job_position')} at {self.state.get_context('company')}. Here is your first question: {first_question}"})
            else:
                return jsonify({'response': "Please provide a valid company name and job position in the format 'Company, Job Position'."})

        if user_input.lower() == "quit":
            self.is_job_position_verified = False
            df = self.evaluate_all_answers()
            self.state.reset()
            return jsonify({'response': "Interview terminated. Thank you for using the Interview Prep Chatbot.", 'quit': True})

        self.collect_user_answer(user_input)
        next_question = self.get_next_question()
        if next_question:
            return jsonify({'response': f"Next question: {next_question}"})
        else:
            df = self.evaluate_all_answers()
            self.state.reset()
            return jsonify({'response': "Interview completed. Thank you for using the Interview Prep Chatbot.", 'quit': True})

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
    return response

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
