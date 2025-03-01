# olabs_chatbot_v4.py
import json
import os
import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from urllib.parse import urljoin
import time
import textwrap
from collections import defaultdict
import speech_recognition as sr
from gtts import gTTS
import pygame.mixer
import tempfile
import keyboard
import re
from quiz_model import QuizModel
from googletrans import Translator, LANGUAGES

# Configure Gemini API
genai.configure(api_key="YOUR API KEY")
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize pygame mixer and translator
pygame.mixer.init()
translator = Translator()

class OLabsChatbot:
    def __init__(self, base_url="https://www.olabs.edu.in/", max_pages=50):
        self.base_url = base_url
        self.max_pages = max_pages
        self.user_progress = self.load_progress()  # Call load_progress after it's defined
        self.content_texts = []
        self.faiss_index = None
        self.topics_map = {}
        self.sections = {}
        self.subjects = []
        self.subtopics = {}
        self.states = ["guidance", "quiz", "roadmap", "general"]
        self.actions = ["topics_map", "sections", "semantic_search"]
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        # Initialize voice-related attributes
        self.voice_enabled = False
        self.first_run = True
        self.current_language = "en"
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        # Load cached data
        self.load_cached_data()
        if not self.content_texts:
            print("Crawling not completed yet. Forcing crawl...")
            self.crawl_olabs()
        self.load_q_table()
        self.query_count = 0
        self.quiz_model = QuizModel()
        self.last_quiz_weak_areas = []
        self.quiz_completed = False
        self.current_topic = None

    def load_progress(self):
        if os.path.exists("user_progress.json"):
            with open("user_progress.json", "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def load_cached_data(self):
        cache_file = "olabs_crawled.json"
        index_file = "faiss_olabs.index"
        
        if os.path.exists(cache_file) and os.path.exists(index_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.content_texts = data["texts"]
                self.topics_map = data["topics"]
            self.faiss_index = faiss.read_index(index_file)
            self.extract_subjects()
            print(f"Loaded cached data: {len(self.content_texts)} texts, {len(self.topics_map)} topics, {len(self.subjects)} subjects")
            if self.voice_enabled:
                self.speak_response(f"Loaded cached data: {len(self.content_texts)} texts, {len(self.topics_map)} topics, {len(self.subjects)} subjects")
        else:
            print("No cached data found. Crawling OLabs...")
            if self.voice_enabled:
                self.speak_response("No cached data found. Crawling OLabs...")
            self.crawl_olabs()
            self.save_cached_data()
    
    # Rest of the class remains unchanged

    def save_cached_data(self):
        with open("olabs_crawled.json", "w", encoding="utf-8") as f:
            json.dump({"texts": self.content_texts, "topics": self.topics_map}, f)
        if self.faiss_index:
            faiss.write_index(self.faiss_index, "faiss_olabs.index")
        print("Cached data and embeddings saved.")
        if self.voice_enabled:
            self.speak_response("Cached data and embeddings saved.")

    def load_q_table(self):
        if os.path.exists("q_table.json"):
            with open("q_table.json", "r") as f:
                loaded = json.load(f)
                for state, actions in loaded.items():
                    self.q_table[state] = np.array(actions)
        print("Q-table loaded.")
        if self.voice_enabled:
            self.speak_response("Q-table loaded.")

    def save_q_table(self):
        with open("q_table.json", "w") as f:
            json.dump({state: list(actions) for state, actions in self.q_table.items()}, f)
        print("Q-table saved.")
        if self.voice_enabled:
            self.speak_response("Q-table saved.")

    def crawl_olabs(self):
        print("Crawling OLabs website for detailed content...")
        if self.voice_enabled:
            self.speak_response("Crawling OLabs website for detailed content...")
        headers = {"User-Agent": "Mozilla/5.0"}
        visited_urls = set()
        to_crawl = [self.base_url]
        self.content_texts = []
        self.topics_map = {}
        
        while to_crawl and len(visited_urls) < self.max_pages:
            url = to_crawl.pop(0)
            if url in visited_urls or not url.startswith(self.base_url):
                continue
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                
                for tag in soup.find_all(["h1", "h2", "h3", "p", "a", "li", "div"]):
                    text = tag.get_text(strip=True)
                    if text and len(text) > 20:
                        if any(keyword in text.lower() for keyword in ["simulation", "experiment", "procedure", "theory", "olabs"]):
                            self.content_texts.append(text)
                        if tag.name in ["h1", "h2", "h3", "a"] and "?" in url:
                            self.topics_map[text.lower()] = url
                
                for link in soup.find_all("a", href=True):
                    href = urljoin(self.base_url, link["href"])
                    if href.startswith(self.base_url) and href not in visited_urls and any(keyword in href.lower() for keyword in ["sim", "exp", "theory"]):
                        to_crawl.append(href)
                
                visited_urls.add(url)
                print(f"Crawled: {url} | Texts: {len(self.content_texts)} | Topics: {len(self.topics_map)}")
                if self.voice_enabled:
                    self.speak_response(f"Crawled: {url}. Texts: {len(self.content_texts)}. Topics: {len(self.topics_map)}")
                time.sleep(1)
                
            except Exception as e:
                print(f"Error crawling {url}: {e}")
                if self.voice_enabled:
                    self.speak_response(f"Error crawling {url}: {e}")
        
        if self.content_texts:
            embeddings = model.encode(self.content_texts, convert_to_numpy=True)
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(embeddings)
            self.extract_subjects()

    def extract_subjects(self):
        self.subjects = []
        self.sections = {}
        for topic, url in self.topics_map.items():
            topic_lower = topic.lower()
            if any(sub in topic_lower for sub in ["science", "theory", "experiment"]) or len(topic.split()) > 3:
                continue
            if topic_lower not in self.subjects:
                self.subjects.append(topic_lower)
                self.sections[topic_lower] = url.split(self.base_url)[-1] if self.base_url in url else "unknown"
        core_subjects = ["physics", "chemistry", "biology", "mathematics"]
        for subject in core_subjects:
            if subject not in self.subjects:
                self.subjects.append(subject)
                self.sections[subject] = f"?sub={self.subjects.index(subject) + 1}"

    def deep_crawl_subject(self, query):
        headers = {"User-Agent": "Mozilla/5.0"}
        visited_urls = set()
        to_crawl = [self.base_url]
        relevant_texts = []
        
        while to_crawl and len(visited_urls) < self.max_pages * 2:
            url = to_crawl.pop(0)
            if url in visited_urls or not url.startswith(self.base_url):
                continue
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                
                for tag in soup.find_all(["h1", "h2", "h3", "p", "a", "li", "div"]):
                    text = tag.get_text(strip=True).lower()
                    if text and len(text) > 5 and any(subject in text for subject in self.subjects):
                        if query.lower() in text:
                            relevant_texts.append(text)
                
                for link in soup.find_all("a", href=True):
                    href = urljoin(self.base_url, link["href"])
                    if href.startswith(self.base_url) and href not in visited_urls:
                        to_crawl.append(href)
                
                visited_urls.add(url)
                time.sleep(1)
                
            except Exception as e:
                print(f"Error deep crawling {url}: {e}")
                if self.voice_enabled:
                    self.speak_response(f"Error deep crawling {url}: {e}")
        
        return relevant_texts

    def semantic_search(self, query, top_k=10):
        if not self.faiss_index:
            print("No embeddings available. Recrawling...")
            if self.voice_enabled:
                self.speak_response("No embeddings available. Recrawling...")
            self.crawl_olabs()
            self.save_cached_data()
            if not self.faiss_index:
                return []
        query_embedding = model.encode([query])
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        return [self.content_texts[idx] for idx in indices[0] if idx != -1]

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        return self.actions[np.argmax(self.q_table[state])]

    def update_q_table(self, state, action, reward, next_state):
        action_idx = self.actions.index(action)
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state][action_idx] += self.alpha * (
            reward + self.gamma * self.q_table[next_state][best_next_action] - self.q_table[state][action_idx]
        )

    def translate_text(self, text, lang):
        if lang == "en" or lang not in LANGUAGES:
            return text
        try:
            return translator.translate(text, dest=lang).text
        except Exception as e:
            print(f"Translation error: {e}")
            if self.voice_enabled:
                self.speak_response(f"Translation error: {e}")
            return text

    def guide_user(self, user_input, action):
        user_input_lower = user_input.lower()
        response = "=== Guidance ===\n"
        
        subject_match = None
        for subject in self.subjects:
            if subject in user_input_lower:
                subject_match = subject
                break
        
        if not subject_match:
            relevant_texts = self.deep_crawl_subject(user_input)
            if relevant_texts:
                response += "Subject-related information found:\n"
                for i, text in enumerate(relevant_texts[:5], 1):
                    wrapped = textwrap.fill(text, width=70, subsequent_indent="    ")
                    response += f"  {i}. {wrapped}\n"
                return self.translate_text(response, self.current_language)
            else:
                response = "=== Response ===\nNo relevant subject content found for your query. Try asking about a subject!"
                return self.translate_text(response, self.current_language)
        
        if action == "topics_map":
            related_content = [text for text in self.content_texts if subject_match in text.lower()]
            response += f"Subject: {subject_match.capitalize()}\n"
            response += f"  - URL: {self.base_url}{self.sections.get(subject_match, 'unknown')}\n"
            response += f"  - Details: Explore simulations and theory for '{subject_match}'.\n"
            if related_content:
                response += f"  - Related Concepts on OLabs for {subject_match}:\n"
                for i, content in enumerate(related_content[:5], 1):
                    wrapped = textwrap.fill(content, width=70, subsequent_indent="      ")
                    response += f"    {i}. {wrapped}\n"
        
        elif action == "sections":
            base_url = f"{self.base_url}{self.sections.get(subject_match, 'unknown')}"
            related_content = [text for text in self.content_texts if subject_match in text.lower()]
            response += f"Subject: {subject_match.capitalize()}\n"
            response += f"  - URL: {base_url}\n"
            if "simulation" in user_input_lower:
                response += f"  - Focus: Access {subject_match} simulations.\n"
            elif "quiz" in user_input_lower:
                response += f"  - Focus: Take {subject_match} quizzes.\n"
            else:
                response += f"  - Includes: Simulations, quizzes, and theory.\n"
            if related_content:
                response += f"  - Related Concepts on OLabs for {subject_match}:\n"
                for i, content in enumerate(related_content[:5], 1):
                    wrapped = textwrap.fill(content, width=70, subsequent_indent="      ")
                    response += f"    {i}. {wrapped}\n"
        
        elif action == "semantic_search":
            results = self.semantic_search(user_input)
            filtered_results = [r for r in results if subject_match in r.lower()]
            if filtered_results:
                response += f"Related {subject_match.capitalize()} Resources:\n"
                for i, res in enumerate(filtered_results[:5], 1):
                    wrapped = textwrap.fill(res, width=70, subsequent_indent="    ")
                    response += f"  {i}. {wrapped}\n"
            else:
                response += f"No specific {subject_match} content found. Try a more specific query!\n"
        
        response += self.translate_text(f"Ask about subjects like '{subject_match}'!", self.current_language)
        return response

    def analyze_quiz(self, user_id, score, weak_areas, action):
        if user_id not in self.user_progress:
            self.user_progress[user_id] = []
        self.user_progress[user_id].append({"score": score, "weak_areas": ", ".join(weak_areas)})
        self.save_progress()
        
        response = "=== Quiz Analysis ===\n"
        response += f"Score: {score}/10\n"
        if weak_areas:
            response += f"Weak Areas: {', '.join(weak_areas)}\n"
            response += "Recommendations:\n"
            weak_lower = [area.lower() for area in weak_areas]
            if action == "topics_map":
                for topic, url in self.topics_map.items():
                    if any(wl in topic.lower() for wl in weak_lower):
                        response += f"  - Resource: '{topic}'\n"
                        response += f"    - URL: {url}\n"
                        response += f"    - Action: Study simulations and theory.\n"
                        break
            elif action == "sections":
                for subject, section in self.sections.items():
                    if any(subject in wl or any(wl in t.lower() for t in self.content_texts if subject in t.lower()) for wl in weak_lower):
                        response += f"  - Resource: {subject.capitalize()}\n"
                        response += f"    - URL: {self.base_url}{section}\n"
                        response += f"    - Action: Focus on relevant content.\n"
                        break
            elif action == "semantic_search":
                results = self.semantic_search(" ".join(weak_areas))
                if results:
                    response += "  - Related Resources:\n"
                    for i, res in enumerate(results[:3], 1):
                        wrapped = textwrap.fill(res, width=70, subsequent_indent="      ")
                        response += f"    {i}. {wrapped}\n"
        else:
            response += "No weak areas identified. Great job!\n"
        
        response += "Press 'W' for Weak Area Roadmap, 'G' for a New Quiz, or 'E' to End Topic.\n"
        return self.translate_text(response, self.current_language)

    def create_roadmap(self, user_id, user_input, action):
        if user_id not in self.user_progress or not self.user_progress[user_id]:
            return self.translate_text("=== Roadmap ===\nTake a quiz first to build your roadmap!\n", self.current_language)
        
        history = self.user_progress[user_id]
        weak_areas = set(entry["weak_areas"].lower() for entry in history)
        response = "=== Your OLabs Learning Roadmap ===\n\n"
        avg_score = sum(entry["score"] for entry in history) / len(history)
        response += f"Average Score: {avg_score:.1f}%\n"
        response += "Focus Areas:\n"
        
        if action == "topics_map":
            for weak in weak_areas:
                response += f"\n  - {weak.capitalize()}:\n"
                related_topics = [(t, url) for t, url in self.topics_map.items() if weak in t]
                related_content = [text for text in self.content_texts if weak in text.lower()]
                if related_topics:
                    for topic, url in related_topics:
                        response += f"    - Resource: '{topic}'\n"
                        response += f"      - URL: {url}\n"
                        response += f"      - Action: Study simulations and theory.\n"
                if related_content:
                    response += f"    - Related Concepts on OLabs:\n"
                    for i, content in enumerate(related_content[:5], 1):
                        wrapped = textwrap.fill(content, width=70, subsequent_indent="        ")
                        response += f"      {i}. {wrapped}\n"
                response += f"    - Next Step: Take a quiz at {self.base_url}quizzes.\n"
        
        elif action == "sections":
            for weak in weak_areas:
                response += f"\n  - {weak.capitalize()}:\n"
                for subject, section in self.sections.items():
                    if subject in weak or any(weak in t.lower() for t in self.content_texts if subject in t.lower()):
                        related_content = [text for text in self.content_texts if subject in text.lower()]
                        response += f"    - Resource: {subject.capitalize()}\n"
                        response += f"      - URL: {self.base_url}{section}\n"
                        response += f"      - Action: Explore related content.\n"
                        if related_content:
                            response += f"      - Related Concepts on OLabs:\n"
                            for i, content in enumerate(related_content[:5], 1):
                                wrapped = textwrap.fill(content, width=70, subsequent_indent="          ")
                                response += f"        {i}. {wrapped}\n"
                        response += f"      - Next Step: Take a quiz at {self.base_url}quizzes.\n"
                        break
        
        elif action == "semantic_search":
            for weak in weak_areas:
                response += f"\n  - {weak.capitalize()}:\n"
                results = self.semantic_search(weak)
                if results:
                    response += f"    - Resources:\n"
                    for i, res in enumerate(results[:5], 1):
                        wrapped = textwrap.fill(res, width=70, subsequent_indent="        ")
                        response += f"      {i}. {wrapped}\n"
                    response += f"      - Next Step: Take a quiz at {self.base_url}quizzes.\n"
        
        response += "\nGeneral Tips:\n"
        response += "  - Practice with simulations daily.\n"
        response += "  - Aim for scores above 85%.\n"
        response += "  - Press 'G' for a new quiz, 'W' for weak area roadmap.\n"
        return self.translate_text(response, self.current_language)

    def create_weak_area_roadmap(self, user_id):
        if not self.last_quiz_weak_areas:
            return self.translate_text("=== Weak Area Roadmap ===\nNo weak areas identified from last quiz. Take a quiz first!\n", self.current_language)
        
        response = "=== Weak Area Roadmap ===\nFocus on these weak concepts from your last quiz:\n"
        for weak in self.last_quiz_weak_areas:
            response += f"\n  - {weak.capitalize()}:\n"
            related_topics = [(t, url) for t, url in self.topics_map.items() if weak.lower() in t.lower()]
            related_content = [text for text in self.content_texts if weak.lower() in text.lower()]
            if related_topics:
                for topic, url in related_topics:
                    response += f"    - Resource: '{topic}'\n"
                    response += f"      - URL: {url}\n"
                    response += f"      - Action: Study simulations and theory.\n"
            if related_content:
                response += f"    - Related Concepts on OLabs:\n"
                for i, content in enumerate(related_content[:5], 1):
                    wrapped = textwrap.fill(content, width=70, subsequent_indent="        ")
                    response += f"      {i}. {wrapped}\n"
            response += f"    - Next Step: Press 'G' to take a quiz on these topics.\n"
        
        response += "\nGeneral Tips:\n"
        response += "  - Practice with simulations daily.\n"
        response += "  - Aim for scores above 85%.\n"
        response += "  - Press 'G' for a new quiz, 'E' to End Topic.\n"
        return self.translate_text(response, self.current_language)

    def recognize_speech(self):
        if not self.voice_enabled:
            return input("Your Query: ")
        with self.microphone as source:
            print("Listening... Speak your query! (Press 'S' to stop)")
            if self.voice_enabled:
                self.speak_response("Listening... Speak your query! Press 'S' to stop.")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=5)
            
        try:
            for lang in ["en-US", "hi-IN", "te-IN", "ml-IN"]:
                try:
                    text = self.recognizer.recognize_google(audio, language=lang)
                    detected_lang = translator.detect(text).lang
                    if detected_lang in LANGUAGES:
                        self.current_language = detected_lang
                    else:
                        self.current_language = "en"
                    print(f"You said ({self.current_language}): {text}")
                    if self.voice_enabled:
                        self.speak_response(f"You said in {self.current_language}: {text}")
                    return text
                except sr.UnknownValueError:
                    continue
            error_msg = "Sorry, I couldn’t understand what you said."
            print(error_msg)
            if self.voice_enabled:
                self.speak_response(error_msg)
            return self.translate_text(error_msg, self.current_language)
        except sr.RequestError as e:
            error_msg = f"Could not request results; {e}"
            print(error_msg)
            if self.voice_enabled:
                self.speak_response(error_msg)
            return self.translate_text(error_msg, self.current_language)

    def preprocess_for_tts(self, text):
        clean_text = re.sub(r'[=*\-_\[\]#]+', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        return clean_text.strip()

    def speak_response(self, text, lang=None):
        if not self.voice_enabled:
            return
        if lang is None:
            lang = self.current_language
        try:
            clean_text = self.preprocess_for_tts(text)
            tts = gTTS(text=clean_text, lang=lang, slow=False)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_path = temp_file.name
            temp_file.close()
            
            print(f"Saving TTS to: {temp_path}")
            tts.save(temp_path)
            
            if os.path.exists(temp_path):
                print(f"Playing: {temp_path}, Size: {os.path.getsize(temp_path)} bytes")
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    if keyboard.is_pressed("q"):
                        pygame.mixer.music.stop()
                        print("Playback stopped (Q pressed).")
                        break
                    pygame.time.Clock().tick(10)
            else:
                print(f"Error: File not found after saving: {temp_path}")
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    print(f"Cleaned up: {temp_path}")
                except Exception as e:
                    print(f"Failed to clean up {temp_path}: {e}")

    def read_context(self, lang=None):
        context = " ".join(self.content_texts[:10])
        if not context.strip():
            response = self.translate_text("No context available to read.", self.current_language)
            print(response)
            if self.voice_enabled:
                self.speak_response(response)
            return response
        print("Reading OLabs context aloud...")
        if self.voice_enabled:
            self.speak_response("Reading OLabs context aloud...")
            self.speak_response(context)
        return self.translate_text("Finished reading context.", self.current_language)

    def suggest_quiz_topics(self):
        if not self.subjects:
            response = self.translate_text("No subjects available. Please wait for crawling to complete.", self.current_language)
            return response, []
        response = "=== Quiz Subjects ===\nPlease select a subject for your quiz:\n"
        for i, topic in enumerate(self.subjects, 1):
            response += f"{i}. {topic.capitalize()}\n"
        response += "Say or type the number or subject name to choose."
        return self.translate_text(response, self.current_language), self.subjects

    def run_and_analyze_quiz(self, user_id, text, topic=None):
        score, weak_concepts = self.quiz_model.run_quiz(text)
        self.last_quiz_weak_areas = weak_concepts
        self.quiz_completed = True
        response = f"=== Quiz Result ===\nSubject: {topic.capitalize() if topic else 'General'}\nScore: {score}/10\n"
        if weak_concepts:
            response += f"Weak Concepts: {', '.join(weak_concepts)}\n"
            response += "Recommendations:\n"
            for wc in weak_concepts:
                for t, url in self.topics_map.items():
                    if wc in t.lower():
                        response += f"  - Study '{t}' at {url}\n"
                        break
        analysis = self.analyze_quiz(user_id, score, weak_concepts, "topics_map")
        return self.translate_text(response + "\n" + analysis, self.current_language)

    def generate_response(self, query, user_id="default_user"):
        print(f"DEBUG: Processing query: {query}")
        if self.voice_enabled:
            self.speak_response(f"Processing query: {query}")
        self.query_count += 1
        query_lower = query.lower()
        state = "general"
        action = "semantic_search"
        
        if "find" in query_lower or "learn" in query_lower or any(s in query_lower for s in self.subjects):
            state = "guidance"
            action = self.choose_action(state)
            response = self.guide_user(query, action)
            print(f"DEBUG: Guidance response generated for state={state}, action={action}")
            if self.voice_enabled:
                self.speak_response(response)
        elif "quiz" in query_lower or "take quiz" in query_lower:
            state = "quiz"
            action = "topics_map"
            if not self.content_texts:
                response = "=== Error ===\nNo content available to generate quiz. Please wait for crawling to complete.\n"
                print(response)
                if self.voice_enabled:
                    self.speak_response(response)
            else:
                topics_response, topics = self.suggest_quiz_topics()
                print(f"Chatbot: {topics_response}")
                if self.voice_enabled:
                    self.speak_response(topics_response)
                print("Listening for subject selection. Say or type the number/subject, or press 'S' to exit.")
                if self.voice_enabled:
                    self.speak_response("Listening for subject selection. Say or type the number or subject, or press 'S' to exit.")
                while True:
                    if keyboard.is_pressed("s"):
                        print("Exiting quiz selection (S pressed).")
                        if self.voice_enabled:
                            self.speak_response("Exiting quiz selection.")
                        return self.translate_text("Quiz selection cancelled.", self.current_language), state, action
                    user_input = self.recognize_speech()
                    if user_input.lower() == "exit":
                        response = "=== Goodbye ===\nGoodbye!"
                        print(response)
                        if self.voice_enabled:
                            self.speak_response(response)
                        return self.translate_text(response, self.current_language), state, action
                    
                    try:
                        choice = int(user_input) - 1
                        if 0 <= choice < len(topics):
                            selected_topic = topics[choice]
                            self.current_topic = selected_topic  # Store the selected topic
                            break
                    except ValueError:
                        if user_input.lower() in [t.lower() for t in topics]:
                            selected_topic = next(t for t in topics if t.lower() == user_input.lower())
                            self.current_topic = selected_topic  # Store the selected topic
                            break
                        print(f"Please say a number (1-{len(topics)}) or a subject name from the list.")
                        if self.voice_enabled:
                            self.speak_response(f"Please say a number from 1 to {len(topics)} or a subject name from the list.")
                
                filtered_text = " ".join([text for text in self.content_texts if selected_topic.lower() in text.lower()])
                if not filtered_text.strip():
                    response = f"=== Error ===\nNo OLabs-specific content available for '{selected_topic}'. Try another subject.\n"
                    print(response)
                    if self.voice_enabled:
                        self.speak_response(response)
                else:
                    while True:
                        response = self.run_and_analyze_quiz(user_id, filtered_text, self.current_topic)
                        print(f"Chatbot: {response}")
                        if self.voice_enabled:
                            self.speak_response(response)
                        while True:
                            print("\nPost-Quiz Options:")
                            print("1. Weak Area Roadmap ('W')")
                            print("2. New Quiz ('G') - Continue with current topic")
                            print("3. End Topic ('E')")
                            if self.voice_enabled:
                                self.speak_response("Post-Quiz Options: Press 'W' for Weak Area Roadmap, 'G' for a New Quiz continuing with current topic, or 'E' to End Topic.")
                            choice = input("Enter your choice (W/G/E): ").lower() if not self.voice_enabled else self.recognize_speech().lower()
                            if choice == "w" or keyboard.is_pressed("w"):
                                roadmap_response = self.create_weak_area_roadmap(user_id)
                                print(f"Chatbot: {roadmap_response}")
                                if self.voice_enabled:
                                    self.speak_response(roadmap_response)
                                continue
                            elif choice == "g" or keyboard.is_pressed("g"):
                                filtered_text = " ".join([text for text in self.content_texts if self.current_topic.lower() in text.lower()])
                                if filtered_text.strip():
                                    response = self.run_and_analyze_quiz(user_id, filtered_text, self.current_topic)
                                    print(f"Chatbot: {response}")
                                    if self.voice_enabled:
                                        self.speak_response(response)
                                    break  # Continue with the quiz loop
                                else:
                                    print(f"Chatbot: No content available for '{self.current_topic}'.")
                                    if self.voice_enabled:
                                        self.speak_response(f"No content available for '{self.current_topic}'.")
                                    break
                            elif choice == "e" or keyboard.is_pressed("e"):
                                print("Ending topic and returning to main menu.")
                                if self.voice_enabled:
                                    self.speak_response("Ending topic and returning to main menu.")
                                return response, state, action
                            else:
                                print("Invalid choice. Please enter 'W', 'G', or 'E'.")
                                if self.voice_enabled:
                                    self.speak_response("Invalid choice. Please say 'W', 'G', or 'E'.")
        elif "roadmap" in query_lower or "plan" in query_lower:
            state = "roadmap"
            action = self.choose_action(state)
            response = self.create_roadmap(user_id, query, action)
            print(f"DEBUG: Roadmap response generated for state={state}, action={action}")
            if self.voice_enabled:
                self.speak_response(response)
        else:
            results = self.semantic_search(query)
            filtered_results = [r for r in results if any(s in r.lower() for s in self.subjects)]
            if filtered_results:
                response = "=== Response ===\nRelated Subject Resources:\n"
                for i, res in enumerate(filtered_results[:5], 1):
                    wrapped = textwrap.fill(res, width=70, subsequent_indent="    ")
                    response += f"  {i}. {wrapped}\n"
            else:
                relevant_texts = self.deep_crawl_subject(query)
                if relevant_texts:
                    response = "=== Response ===\nSubject-related information found:\n"
                    for i, text in enumerate(relevant_texts[:5], 1):
                        wrapped = textwrap.fill(text, width=70, subsequent_indent="    ")
                        response += f"  {i}. {wrapped}\n"
                else:
                    response = "=== Response ===\nNo relevant subject content found. Ask about a subject!\n"
            print(f"DEBUG: General response generated with {len(filtered_results)} results")
            if self.voice_enabled:
                self.speak_response(response)
        
        if self.query_count % 7 == 0 and "Did this help?" not in response:
            response += self.translate_text("Did this help? (yes/no)", self.current_language)
        return response, state, action

    def handle_feedback(self, feedback, state, action):
        reward = 1 if feedback.lower() == "yes" else -1 if feedback.lower() == "no" else 0
        next_state = "general"
        if reward != 0:
            self.update_q_table(state, action, reward, next_state)
            self.save_q_table()
        response = self.translate_text("Feedback processed. What’s next?", self.current_language)
        return response

if __name__ == "__main__":
    chatbot = OLabsChatbot(max_pages=50)
    awaiting_feedback = False
    last_state = None
    last_action = None
    
    intro_message = "I am your OLabs Assistant, here to guide you through science learning!\nPlease choose your input mode: Type 'text' or 'voice'."
    print(f"Chatbot: {intro_message}")
    chatbot.speak_response(intro_message)  # Always speak intro message
    
    while chatbot.first_run:
        initial_input = input("Input Mode (type 'text' or 'voice'): ").lower() if not chatbot.voice_enabled else chatbot.recognize_speech().lower()
        if initial_input == "text":
            chatbot.voice_enabled = False
            print("Text input mode selected.")
            chatbot.speak_response("Text input mode selected.")
            chatbot.first_run = False
        elif initial_input == "voice":
            chatbot.voice_enabled = True
            print("Voice input mode selected. Say your query!")
            chatbot.speak_response("Voice input mode selected. Say your query!")
            chatbot.first_run = False
        else:
            print("Invalid input. Please type or say 'text' or 'voice'.")
            chatbot.speak_response("Invalid input. Please say 'text' or 'voice'.")
    
    while True:
        menu = "\n=== OLabs Chatbot Options ===\n"
        menu += "1. Enter Query (Text/Voice based on mode)\n"
        menu += "2. Toggle Voice Mode ('V'): Switch between text and voice\n"
        menu += "3. Next Input ('N'): Move to next query\n"
        if chatbot.quiz_completed:
            menu += "4. Weak Area Roadmap ('W'): View roadmap for weak areas\n"
            menu += "5. Next Quiz ('G'): Take a quiz on current topic\n"
        menu += "6. Exit ('S' or 'exit'): Stop the chatbot\n"
        menu += "============================"
        print(menu)
        if chatbot.voice_enabled:
            chatbot.speak_response(menu.replace('\n', '. '))
        
        if not chatbot.voice_enabled:
            user_input = input("Your Query: ")
        else:
            user_input = chatbot.recognize_speech()
        
        if user_input.lower() == "exit" or keyboard.is_pressed("s"):
            response = "=== Goodbye ===\nGoodbye!"
            print(response)
            if chatbot.voice_enabled:
                chatbot.speak_response(response)
            break
        
        if keyboard.is_pressed("v"):
            chatbot.voice_enabled = not chatbot.voice_enabled
            mode_msg = "Voice input mode selected. Say your query!" if chatbot.voice_enabled else "Text input mode selected."
            print(mode_msg)
            if chatbot.voice_enabled:
                chatbot.speak_response(mode_msg)
            continue
        
        if keyboard.is_pressed("n"):
            print("Moving to next input.")
            if chatbot.voice_enabled:
                chatbot.speak_response("Moving to next input.")
            continue
        
        if (keyboard.is_pressed("w") or keyboard.is_pressed("g")) and not chatbot.quiz_completed:
            msg = "Please complete a quiz first to access Weak Area Roadmap or Next Quiz."
            print(msg)
            if chatbot.voice_enabled:
                chatbot.speak_response(msg)
            continue
        
        if keyboard.is_pressed("w") and chatbot.quiz_completed:
            response = chatbot.create_weak_area_roadmap("default_user")
            print(f"Chatbot: {response}")
            if chatbot.voice_enabled:
                chatbot.speak_response(response)
        elif keyboard.is_pressed("g") and chatbot.quiz_completed:
            filtered_text = " ".join([text for text in chatbot.content_texts if chatbot.current_topic.lower() in text.lower()])
            if filtered_text.strip():
                response = chatbot.run_and_analyze_quiz("default_user", filtered_text, chatbot.current_topic)
                print(f"Chatbot: {response}")
                if chatbot.voice_enabled:
                    chatbot.speak_response(response)
            continue
        else:
            response, state, action = chatbot.generate_response(user_input, "default_user")
            print(f"Chatbot: {response}")
            if chatbot.voice_enabled:
                chatbot.speak_response(response)
        
        if "Did this help?" in response:
            feedback_prompt = "Please provide feedback. Say or type 'yes' or 'no'."
            print(feedback_prompt)
            if chatbot.voice_enabled:
                chatbot.speak_response(feedback_prompt)
            feedback = input("Feedback (type 'yes' or 'no'): ") if not chatbot.voice_enabled else chatbot.recognize_speech()
            response = chatbot.handle_feedback(feedback, state, action)
            print(f"Chatbot: {response}")
            if chatbot.voice_enabled:
                chatbot.speak_response(response)
            last_state, last_action = None, None
        else:
            last_state, last_action = state, action
        
        post_convo = "\nPost-Conversation Options:\n1. Next Input ('N'): Move to next query\n2. Read Context ('R'): Hear crawled context"
        print(post_convo)
        if chatbot.voice_enabled:
            chatbot.speak_response(post_convo.replace('\n', '. '))
        choice_prompt = "Enter your choice (N/R): "
        print(choice_prompt, end='')
        if chatbot.voice_enabled:
            chatbot.speak_response("Say 'N' for next input or 'R' to read context.")
        choice = input() if not chatbot.voice_enabled else chatbot.recognize_speech().lower()
        if choice == "n" or keyboard.is_pressed("n"):
            print("Moving to next input.")
            if chatbot.voice_enabled:
                chatbot.speak_response("Moving to next input.")
            continue
        elif choice == "r" or keyboard.is_pressed("r"):
            context_response = chatbot.read_context()
            print(f"Chatbot: {context_response}")
            if chatbot.voice_enabled:
                chatbot.speak_response(context_response)
            continue
        else:
            print("Invalid choice. Moving to next input.")
            if chatbot.voice_enabled:
                chatbot.speak_response("Invalid choice. Moving to next input.")
        
        mode_prompt = "\nPlease choose your next input mode: Type 'text' or 'voice'"
        print(mode_prompt)
        if chatbot.voice_enabled:
            chatbot.speak_response("Please choose your next input mode. Say 'text' or 'voice'.")
        mode_input = input("Type 'text' or 'voice': ").lower() if not chatbot.voice_enabled else chatbot.recognize_speech().lower()
        if mode_input == "text":
            chatbot.voice_enabled = False
            print("Text input mode selected.")
            if chatbot.voice_enabled:
                chatbot.speak_response("Text input mode selected.")
        elif mode_input == "voice":
            chatbot.voice_enabled = True
            print("Voice input mode selected. Say your query!")
            if chatbot.voice_enabled:
                chatbot.speak_response("Voice input mode selected. Say your query!")
        else:
            print("Invalid input. Defaulting to text mode.")
            if chatbot.voice_enabled:
                chatbot.speak_response("Invalid input. Defaulting to text mode.")
            chatbot.voice_enabled = False
