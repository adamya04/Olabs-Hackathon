# Install required packages (run this once in a separate cell if not already installed)
!pip install flask google-generativeai googletrans==3.1.0a0 numpy

# Import libraries and define the Flask app
from flask import Flask, request, render_template_string
import random
import numpy as np
from collections import defaultdict
import google.generativeai as genai
from googletrans import Translator
import threading
import time

app = Flask(__name__)

# RL parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.5
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
NUM_DIFFICULTIES = len(DIFFICULTY_LEVELS)
NUM_PERFORMANCE_BUCKETS = 3
NUM_MASTERY_LEVELS = 3
STATE_SPACE_SIZE = NUM_DIFFICULTIES * NUM_PERFORMANCE_BUCKETS * NUM_MASTERY_LEVELS
NUM_ACTIONS = 3

# Initialize Q-table with bias
Q = np.zeros((STATE_SPACE_SIZE, NUM_ACTIONS))
for difficulty in range(NUM_DIFFICULTIES):
    for performance in range(NUM_PERFORMANCE_BUCKETS):
        for mastery in range(NUM_MASTERY_LEVELS):
            state_idx = difficulty * (NUM_PERFORMANCE_BUCKETS * NUM_MASTERY_LEVELS) + performance * NUM_MASTERY_LEVELS + mastery
            if performance == 0:
                Q[state_idx][0] = 0.5
            elif performance == 2:
                Q[state_idx][2] = 0.5

concept_mastery = defaultdict(list)

API_KEY = "AIzaSyCzmpRdo3Qo7VT_eb5R9GJgey_HnO5xAc0"
genai.configure(api_key=API_KEY)
translator = Translator()

# Global state
current_difficulty = 1  # Start at medium
previous_score = 5
quiz_state = {
    'questions': [],
    'sample_answers_or_options': [],
    'concepts': [],
    'is_mcq_flags': [],
    'user_answers': [],
    'total_score': None,
    'feedback': '',
    'text': ''
}

def setup_model():
    return genai.GenerativeModel('gemini-1.5-flash')

def get_state_index(difficulty, performance, mastery):
    return difficulty * (NUM_PERFORMANCE_BUCKETS * NUM_MASTERY_LEVELS) + performance * NUM_MASTERY_LEVELS + mastery

def get_performance_bucket(score):
    if score <= 3: return 0
    elif score <= 7: return 1
    else: return 2

def get_mastery_level(concepts_of_interest):
    if not concepts_of_interest: return 1
    total_correct = sum(sum(concept_mastery.get(concept, [])) for concept in concepts_of_interest)
    total_questions = sum(len(concept_mastery.get(concept, [])) for concept in concepts_of_interest)
    if total_questions == 0: return 1
    accuracy = total_correct / total_questions
    return 0 if accuracy < 0.4 else 1 if accuracy < 0.7 else 2

def choose_action(state_index, epsilon, score):
    if score < 5: return 0
    elif score >= 8: return 2
    else: return 1

def update_q_table(state_index, action, reward, next_state_index, learning_rate):
    old_value = Q[state_index][action]
    next_max = np.max(Q[next_state_index])
    Q[state_index][action] = (1 - learning_rate) * old_value + learning_rate * (reward + DISCOUNT_FACTOR * next_max)

def calculate_reward(score, previous_score, difficulty):
    base_reward = 2 if score >= 8 else 1 if score >= 5 else 0 if score >= 3 else -2
    improvement_reward = 1 if score > previous_score else 0 if score == previous_score else -1
    difficulty_bonus = difficulty * 0.5 if score >= 5 else -difficulty * 0.5
    return base_reward + improvement_reward + difficulty_bonus

def adjust_difficulty(current_difficulty, action):
    if action == 0: return max(0, current_difficulty - 1)
    elif action == 1: return current_difficulty
    elif action == 2: return min(NUM_DIFFICULTIES - 1, current_difficulty + 1)

def generate_questions(model, difficulty, text):
    difficulty_desc = {"easy": "simple facts", "medium": "processes", "hard": "complex applications"}
    prompt = f"""
    Generate 10 unique questions strictly based on the following text at {DIFFICULTY_LEVELS[difficulty]} difficulty - {difficulty_desc[DIFFICULTY_LEVELS[difficulty]]}:
    - 5 open-ended questions requiring text-based responses
    - 5 multiple-choice questions (MCQs) with 4 options (A, B, C, D)
    All questions must be directly relevant to the text content.
    For each:
    1. Include a concept label
    2. For open-ended: Provide a concise sample answer
    3. For MCQs: Provide 4 options and the correct answer
    Format exactly as:
    Q1. [Concept: concept_name] Open-ended question?
    Sample Answer: [sample_answer]
    Q6. [Concept: concept_name] MCQ question?
    A. [option_a]
    B. [option_b]
    C. [option_c]
    D. [option_d]
    Correct: [correct_letter]
    Text: {text}
    """
    try:
        response = model.generate_content(prompt)
        return parse_quiz_response(response.text)
    except Exception as e:
        print(f"Error generating questions: {e}")
        return ["Backup question"] * 10, ["Backup answer"] * 5 + [({"A": "A", "B": "B", "C": "C", "D": "D"}, "A")] * 5, ["general"] * 10, [False] * 5 + [True] * 5

def parse_quiz_response(response_text):
    questions, sample_answers_or_options, concepts, is_mcq_flags = [], [], [], []
    question_blocks, current_block = [], []
    for line in response_text.split('\n'):
        if line.strip() and (line.strip().startswith('Q') and any(char.isdigit() for char in line[:3])):
            if current_block:
                question_blocks.append('\n'.join(current_block))
                current_block = []
        if line.strip():
            current_block.append(line)
    if current_block:
        question_blocks.append('\n'.join(current_block))
    
    for block in question_blocks:
        if not block.strip(): continue
        lines = block.strip().split('\n')
        q_line = lines[0]
        concept_start = q_line.find('[Concept:')
        concept_end = q_line.find(']', concept_start) if concept_start != -1 else -1
        concept = q_line[concept_start+9:concept_end].strip() if concept_start != -1 and concept_end != -1 else "general"
        question = q_line[concept_end+1:].strip() if concept_start != -1 and concept_end != -1 else q_line.split('.', 1)[1].strip() if '.' in q_line else q_line
        
        if "Sample Answer:" in block:
            sample_answer = next((line.split('Sample Answer:')[1].strip() for line in lines[1:] if 'Sample Answer:' in line), "No answer provided")
            questions.append(question)
            sample_answers_or_options.append(sample_answer)
            concepts.append(concept)
            is_mcq_flags.append(False)
        elif "Correct:" in block:
            options, correct_answer = {}, None
            for line in lines[1:]:
                if line.startswith(('A.', 'B.', 'C.', 'D.')):
                    opt_letter, opt_text = line[0], line[2:].strip()
                    options[opt_letter] = opt_text
                elif 'Correct:' in line:
                    correct_answer = line.split('Correct:')[1].strip()
            questions.append(question)
            sample_answers_or_options.append((options, correct_answer))
            concepts.append(concept)
            is_mcq_flags.append(True)
    
    while len(questions) < 10:
        questions.append(f"Backup question {len(questions)+1}")
        sample_answers_or_options.append("Backup answer" if len(questions) < 5 else ({"A": "A", "B": "B", "C": "C", "D": "D"}, "A"))
        concepts.append("general")
        is_mcq_flags.append(False if len(questions) < 5 else True)
    return questions[:10], sample_answers_or_options[:10], concepts[:10], is_mcq_flags[:10]

def evaluate_text_answer(model, question, user_answer, sample_answer, concept):
    try:
        detected_lang = translator.detect(user_answer).lang
        if detected_lang == 'hi':
            user_answer = translator.translate(user_answer, src='hi', dest='en').text
    except Exception as e:
        print(f"Translation error: {e}")

    prompt = f"""
    Evaluate this user answer for similarity to the sample answer, based on the concept '{concept}'.
    Score as:
    - 1 mark if fully similar (nearly identical meaning)
    - 0.5 marks if partially similar (some overlap in meaning)
    - 0 marks if not similar (no meaningful match)
    Provide brief feedback.
    
    Question: {question}
    User Answer: {user_answer}
    Sample Answer: {sample_answer}
    
    Format:
    Score: [number]
    Feedback: [text]
    """
    try:
        response = model.generate_content(prompt)
        response_text = response.text
        score_line = next((line for line in response_text.split('\n') if line.startswith('Score:')), 'Score: 0')
        feedback_line = next((line for line in response_text.split('\n') if line.startswith('Feedback:')), 'Feedback: No feedback provided')
        return float(score_line.split(':')[1].strip()), feedback_line.split(':')[1].strip()
    except Exception as e:
        print(f"Error evaluating text answer: {e}")
        return 0, "Evaluation failed"

def evaluate_mcq_answer(user_answer, correct_answer):
    return 1 if user_answer.upper() == correct_answer.upper() else 0

# HTML Templates with Enhanced Styling
HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Adaptive Quiz System</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #74ebd5, #acb6e5);
            margin: 0;
            padding: 0;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            width: 90%;
            max-width: 600px;
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 20px;
            font-weight: 600;
        }
        p {
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 30px;
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 15px;
            border: 2px solid #ecf0f1;
            border-radius: 10px;
            resize: none;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        textarea:focus {
            border-color: #3498db;
            outline: none;
        }
        button {
            display: block;
            width: 100%;
            padding: 15px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s, transform 0.2s;
        }
        button:hover {
            background: #2980b9;
            transform: translateY(-2px);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 Adaptive Quiz System</h1>
        <p>Enter your text below to generate a custom quiz!</p>
        <form method="POST" action="/quiz">
            <textarea name="text" placeholder="Type or paste your text here..." required></textarea>
            <button type="submit">Generate Quiz</button>
        </form>
    </div>
</body>
</html>
"""

QUIZ_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Adaptive Quiz</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #74ebd5, #acb6e5);
            margin: 0;
            padding: 40px 20px;
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            max-width: 800px;
            margin: 0 auto;
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }
        .difficulty {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
            font-size: 18px;
        }
        .question {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            transition: transform 0.2s;
        }
        .question:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }
        .question p {
            color: #34495e;
            margin: 0 0 15px 0;
            font-weight: 600;
        }
        .options label {
            display: block;
            padding: 10px;
            background: #ecf0f1;
            margin: 5px 0;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.3s;
        }
        .options input[type="radio"] {
            margin-right: 10px;
        }
        .options label:hover {
            background: #dfe6e9;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #ecf0f1;
            border-radius: 10px;
            font-size: 16px;
            resize: vertical;
            transition: border-color 0.3s;
        }
        textarea:focus {
            border-color: #3498db;
            outline: none;
        }
        button {
            display: block;
            width: 100%;
            padding: 15px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s, transform 0.2s;
        }
        button:hover {
            background: #2980b9;
            transform: translateY(-2px);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎓 Adaptive Quiz</h1>
        <p class="difficulty">Difficulty: {{ difficulty }}</p>
        <form method="POST" action="/submit">
            {% for i in range(questions|length) %}
                <div class="question">
                    <p>Q{{ i + 1 }}. [{{ concepts[i] }}] {{ questions[i] }}</p>
                    {% if is_mcq_flags[i] %}
                        <div class="options">
                            {% set options, correct = sample_answers_or_options[i] %}
                            <label><input type="radio" name="answer_{{ i }}" value="A" required> A. {{ options['A'] }}</label>
                            <label><input type="radio" name="answer_{{ i }}" value="B"> B. {{ options['B'] }}</label>
                            <label><input type="radio" name="answer_{{ i }}" value="C"> C. {{ options['C'] }}</label>
                            <label><input type="radio" name="answer_{{ i }}" value="D"> D. {{ options['D'] }}</label>
                        </div>
                    {% else %}
                        <textarea name="answer_{{ i }}" placeholder="Type your answer here..." required></textarea>
                    {% endif %}
                </div>
            {% endfor %}
            <button type="submit">Submit Answers</button>
        </form>
    </div>
</body>
</html>
"""

RESULT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Quiz Result</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #74ebd5, #acb6e5);
            margin: 0;
            padding: 40px 20px;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            max-width: 800px;
            width: 90%;
            text-align: center;
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        h1 {
            color: #2c3e50;
            margin-bottom: 20px;
        }
        .score {
            font-size: 36px;
            color: #3498db;
            font-weight: 600;
            margin-bottom: 20px;
        }
        .feedback {
            color: #34495e;
            font-size: 16px;
            text-align: left;
            margin-bottom: 20px;
            background: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
        }
        .feedback h3 {
            color: #e74c3c;
            margin-top: 0;
        }
        .feedback-item {
            margin-bottom: 15px;
            padding: 10px;
            border-left: 4px solid #e74c3c;
            background: #fff;
        }
        .feedback-item p {
            margin: 5px 0;
        }
        .next-difficulty {
            color: #34495e;
            font-size: 18px;
            margin-bottom: 30px;
        }
        button {
            padding: 15px 30px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s, transform 0.2s;
        }
        button:hover {
            background: #2980b9;
            transform: translateY(-2px);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎉 Quiz Result</h1>
        <div class="score">{{ score }} / 10</div>
        <div class="feedback">
            {{ feedback|safe }}
        </div>
        <p class="next-difficulty">Next Difficulty: {{ next_difficulty }}</p>
        <form method="GET" action="/">
            <button type="submit">Try Another Quiz</button>
        </form>
    </div>
</body>
</html>
"""

# Routes
@app.route('/', methods=['GET'])
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route('/quiz', methods=['POST'])
def quiz():
    global quiz_state, current_difficulty
    text = request.form['text']
    if not text.strip():
        return "Text cannot be empty", 400
    
    model = setup_model()
    questions, sample_answers_or_options, concepts, is_mcq_flags = generate_questions(model, current_difficulty, text)
    
    quiz_state['text'] = text
    quiz_state['questions'] = questions
    quiz_state['sample_answers_or_options'] = sample_answers_or_options
    quiz_state['concepts'] = concepts
    quiz_state['is_mcq_flags'] = is_mcq_flags
    quiz_state['user_answers'] = [''] * len(questions)
    quiz_state['total_score'] = None
    quiz_state['feedback'] = ''

    return render_template_string(QUIZ_TEMPLATE, 
        difficulty=DIFFICULTY_LEVELS[current_difficulty],
        questions=quiz_state['questions'],
        sample_answers_or_options=quiz_state['sample_answers_or_options'],
        concepts=quiz_state['concepts'],
        is_mcq_flags=quiz_state['is_mcq_flags']
    )

@app.route('/submit', methods=['POST'])
def submit():
    global quiz_state, current_difficulty, previous_score
    model = setup_model()
    
    for i in range(len(quiz_state['questions'])):
        quiz_state['user_answers'][i] = request.form.get(f'answer_{i}', '')

    total_score = 0
    concept_results = defaultdict(list)
    weak_areas = defaultdict(list)

    # Store detailed feedback for weak areas
    feedback_details = []

    for i, (question, user_answer, answer_data, concept, is_mcq) in enumerate(zip(
        quiz_state['questions'], quiz_state['user_answers'], quiz_state['sample_answers_or_options'],
        quiz_state['concepts'], quiz_state['is_mcq_flags']
    )):
        if is_mcq:
            _, correct_answer = answer_data
            score = evaluate_mcq_answer(user_answer, correct_answer)
            feedback = "Correct" if score == 1 else f"Incorrect (Correct answer: {correct_answer})"
        else:
            sample_answer = answer_data
            score, feedback = evaluate_text_answer(model, question, user_answer, sample_answer, concept)
        
        concept_results[concept].append(score)
        total_score += score
        
        # If score is below threshold, add detailed feedback
        if score < (1 if is_mcq else 0.5):
            weak_areas[concept].append(feedback)
            feedback_details.append({
                'concept': concept,
                'question': question,
                'user_answer': user_answer,
                'sample_answer': sample_answer if not is_mcq else f"Correct: {correct_answer}",
                'feedback': feedback,
                'score': score
            })

    for concept, results in concept_results.items():
        concept_mastery[concept].extend(results)

    performance_bucket = get_performance_bucket(previous_score)
    mastery_level = get_mastery_level(list(concept_results.keys()))
    current_state_index = get_state_index(current_difficulty, performance_bucket, mastery_level)
    reward = calculate_reward(total_score, previous_score, current_difficulty)
    action = choose_action(current_state_index, EPSILON, total_score)
    next_difficulty = adjust_difficulty(current_difficulty, action)
    next_state_index = get_state_index(next_difficulty, get_performance_bucket(total_score), get_mastery_level(list(concept_results.keys())))
    update_q_table(current_state_index, action, reward, next_state_index, LEARNING_RATE)

    # Construct detailed feedback
    feedback_para = "<h3>Performance Summary</h3>"
    if total_score >= 8:
        feedback_para += "<p>Excellent work! You scored high and demonstrated strong understanding across most concepts.</p>"
    elif total_score >= 5:
        feedback_para += "<p>Good effort! You have a solid grasp of many concepts, with some areas to improve.</p>"
    else:
        feedback_para += "<p>Keep practicing! There are several areas where you can improve your understanding.</p>"

    if weak_areas:
        feedback_para += "<h3>Areas for Improvement</h3>"
        for detail in feedback_details:
            feedback_para += f"""
            <div class="feedback-item">
                <p><strong>Concept:</strong> {detail['concept']}</p>
                <p><strong>Question:</strong> {detail['question']}</p>
                <p><strong>Your Answer:</strong> {detail['user_answer']}</p>
                <p><strong>Expected Answer:</strong> {detail['sample_answer']}</p>
                <p><strong>Score:</strong> {detail['score']}</p>
                <p><strong>Feedback:</strong> {detail['feedback']}</p>
            </div>
            """
    else:
        feedback_para += "<p>Congratulations! You performed well across all concepts with no significant weaknesses identified.</p>"

    quiz_state['total_score'] = total_score
    quiz_state['feedback'] = feedback_para
    previous_score = total_score
    current_difficulty = next_difficulty

    return render_template_string(RESULT_TEMPLATE, 
        score=total_score,
        feedback=feedback_para,
        next_difficulty=DIFFICULTY_LEVELS[next_difficulty]
    )

# Function to run Flask in a separate thread
def run_flask():
    app.run(debug=True, use_reloader=False, port=5000)

# Start Flask in a background thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True  # Daemonize so it stops when notebook stops
flask_thread.start()

# Give the server a moment to start
time.sleep(1)

print("Flask app is running at http://127.0.0.1:5000")
print("Open this URL in your browser to use the quiz app!")