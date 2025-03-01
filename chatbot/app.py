from flask import Flask, render_template, request, jsonify
from olabs_chatbot_v4 import OLabsChatbot
import json

app = Flask(__name__)

# Initialize the chatbot
chatbot = OLabsChatbot(max_pages=50)
chatbot.voice_enabled = False  # Default to text mode for web interface

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '').strip()
    user_id = data.get('user_id', 'default_user')
    
    if not user_input:
        return jsonify({'response': 'Please enter a query.'})
    
    if user_input.lower() == 'exit':
        return jsonify({'response': '=== Goodbye ===\nGoodbye!'})
    
    # Generate response from chatbot
    response, state, action = chatbot.generate_response(user_input, user_id)
    
    # Handle quiz subject selection explicitly
    if "quiz" in user_input.lower():
        if "suggest_quiz_topics" in response:  # Initial quiz prompt
            return jsonify({
                'response': response,
                'state': state,
                'action': action,
                'quiz_selection': True
            })
        elif chatbot.current_topic:  # Quiz running with selected topic
            return jsonify({
                'response': response,
                'state': state,
                'action': action,
                'quiz_selection': False
            })
    
    feedback_required = "Did this help?" in response
    return jsonify({
        'response': response,
        'state': state,
        'action': action,
        'feedback_required': feedback_required,
        'quiz_selection': False
    })

@app.route('/select_quiz_topic', methods=['POST'])
def select_quiz_topic():
    data = request.get_json()
    user_input = data.get('selection', '').strip()
    user_id = data.get('user_id', 'default_user')
    
    topics = chatbot.subjects
    try:
        choice = int(user_input) - 1
        if 0 <= choice < len(topics):
            chatbot.current_topic = topics[choice]
            filtered_text = " ".join([text for text in chatbot.content_texts if chatbot.current_topic.lower() in text.lower()])
            if not filtered_text.strip():
                return jsonify({'response': f"No content available for '{chatbot.current_topic}'."})
            response = chatbot.run_and_analyze_quiz(user_id, filtered_text, chatbot.current_topic)
            return jsonify({'response': response})
    except ValueError:
        if user_input.lower() in [t.lower() for t in topics]:
            chatbot.current_topic = next(t for t in topics if t.lower() == user_input.lower())
            filtered_text = " ".join([text for text in chatbot.content_texts if chatbot.current_topic.lower() in text.lower()])
            if not filtered_text.strip():
                return jsonify({'response': f"No content available for '{chatbot.current_topic}'."})
            response = chatbot.run_and_analyze_quiz(user_id, filtered_text, chatbot.current_topic)
            return jsonify({'response': response})
    
    return jsonify({'response': 'Invalid selection. Please choose a number or subject from the list.'})

# Other routes remain unchanged
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    feedback = data.get('feedback', '').strip()
    state = data.get('state', '')
    action = data.get('action', '')
    response = chatbot.handle_feedback(feedback, state, action)
    return jsonify({'response': response})

@app.route('/toggle_voice', methods=['POST'])
def toggle_voice():
    chatbot.voice_enabled = not chatbot.voice_enabled
    mode = "Voice mode enabled (Note: Voice not fully supported in web UI)" if chatbot.voice_enabled else "Text mode enabled"
    return jsonify({'response': mode})

@app.route('/weak_roadmap', methods=['POST'])
def weak_roadmap():
    if not chatbot.quiz_completed:
        return jsonify({'response': 'Please complete a quiz first to access Weak Area Roadmap.'})
    response = chatbot.create_weak_area_roadmap('default_user')
    return jsonify({'response': response})

@app.route('/next_quiz', methods=['POST'])
def next_quiz():
    if not chatbot.quiz_completed or not chatbot.current_topic:
        return jsonify({'response': 'Please complete a quiz first or select a topic.'})
    filtered_text = " ".join([text for text in chatbot.content_texts if chatbot.current_topic.lower() in text.lower()])
    if not filtered_text.strip():
        return jsonify({'response': f"No content available for '{chatbot.current_topic}'."})
    response = chatbot.run_and_analyze_quiz('default_user', filtered_text, chatbot.current_topic)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)