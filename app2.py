from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

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

    def update_context(self, key, value):
        self.context[key] = value

    def get_context(self, key):
        return self.context.get(key, None)

    def add_candidate_response(self, question, response):
        self.candidate_responses.append({'question': question, 'response': response})

    def reset(self):
        self.context = {}
        self.candidate_responses = []

# Define chatbot
class InterviewBot:
    def __init__(self, dataset):
        self.state = ConversationState()
        self.dataset = dataset
        self.is_job_position_verified = False
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(self.dataset['Answers'])

    def nlu(self, user_input):
        doc = nlp(user_input)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

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
            self.start_interview(company, job_position)
            return True
        else:
            return False

    def evaluate_answer(self, user_input, question):
        expected_answer = self.get_answer_for_question(question)
        responses = [user_input, expected_answer]
        tfidf_matrix = self.vectorizer.transform(responses)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).flatten()[0]
        performance = categorize_performance(similarity)
        return similarity, performance

    def respond(self, user_input, company=None, domain=None):
        if company and domain and not self.is_job_position_verified:
            if self.verify_job_position(company, domain):
                self.is_job_position_verified = True
                return f"Got it. Starting interview for {self.state.get_context('job_position')} at {self.state.get_context('company')}. Here is your first question: {self.get_next_question()}"
            else:
                return "Please provide a valid company name and job position in the format 'Company, Job Position'."

        if user_input.lower() == "quit":
            self.is_job_position_verified = False
            self.state.reset()
            return "Interview terminated. Thank you for using the Interview Prep Chatbot."

        current_question = self.state.get_context("questions")[self.state.get_context("current_question_index") - 1]
        self.state.add_candidate_response(current_question, user_input)

        # Evaluate the answer
        similarity, performance = self.evaluate_answer(user_input, current_question)
        print(f"Similarity score for the answer: {similarity:.2f}")

        next_question = self.get_next_question()
        if next_question:
            return f"{performance} Next question: {next_question}"
        else:
            return "Thank you for completing the interview. We will now evaluate your responses."

def categorize_performance(similarity):
    if similarity <= 0.2:
        return "Poor performance."
    elif similarity <= 0.4:
        return "Needs improvement."
    elif similarity <= 0.6:
        return "Better."
    elif similarity <= 0.8:
        return "Good."
    else:
        return "Excellent."

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
