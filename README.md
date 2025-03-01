Welcome to the **Question Generator** and the **chatbot** project! This repository contains an innovative system that leverages reinforcement learning (RL) to generate questions, dynamically updates based on user inputs, supports multiple languages, and integrates a custom large language model (LLM) built using web scraping. Additionally, we incorporate speech capabilities using OpenAI's Whisper for text-to-speech generation.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [How It Works](#how-it-works)
   - [Question Generator with Reinforcement Learning](#question-generator-with-reinforcement-learning)
   - [Dynamic Updates Based on User Input](#dynamic-updates-based-on-user-input)
   - [Multilingual Support](#multilingual-support)
   - [Custom LLM with Web Scraping](#custom-llm-with-web-scraping)
   - [Speech Integration with Whisper](#speech-integration-with-whisper)
6. [Contributing](#contributing)
7. [License](#license)

## Project Overview
This project is designed to develop a sophisticated and adaptive question-generation system that evolves dynamically through user interaction. At its core, the system integrates cutting-edge technologies—reinforcement learning (RL), a custom-built large language model (LLM) trained on diverse web-scraped data, and advanced speech processing capabilities—to create a highly interactive and personalized user experience. The goal is to craft a tool that not only generates contextually relevant questions but also learns from each interaction, tailoring its behavior to individual user preferences over time.

By leveraging reinforcement learning, the system optimizes its question-generation process through a reward-based mechanism, ensuring that the questions it produces are engaging, meaningful, and aligned with the user’s interests. The custom LLM, constructed from a rich dataset harvested through web scraping, empowers the system with a broad knowledge base and the ability to generate creative, nuanced, and context-aware responses. This LLM is further enhanced by its capacity to incorporate real-time user inputs, allowing it to refine its understanding and improve its outputs continuously.

## Features
- **Reinforcement Learning Question Generator**: Generates contextually relevant questions using an RL-based approach.
- **Dynamic Updates**: Adapts and refines outputs based on user responses in real-time.
- **Multilingual Support**: Accepts inputs and provides outputs in multiple languages.
- **Custom LLM**: Built from scratch using web-scraped data for unique and diverse responses.
- **Speech Capabilities**: Uses Whisper to convert generated text into speech dynamically.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/question-generator.git
   cd question-generator
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Whisper for speech generation (requires additional setup):
   - Follow the instructions in the [Whisper GitHub repository](https://github.com/openai/whisper).
4. (Optional) Configure language models or pretrained weights for the custom LLM.

## Usage
1. Run the main script:
   ```bash
   python main.py
   ```
2. Input your responses via text or speech (if enabled).
3. Select your preferred language from the supported list.
4. Receive dynamically generated questions and hear them via Whisper-generated audio.

Example:
```bash
$ python main.py
Enter your input: "I enjoy hiking and reading."
Preferred language: English
Output: "What kind of books do you read while hiking?" (spoken via Whisper)
```

## How It Works

### Question Generator with Reinforcement Learning
- The core of the system uses a reinforcement learning model integrated with gemini api to generate questions.
- The RL agent is trained to maximize a reward function based on user engagement and question relevance.
- Over time, the model learns to ask better questions tailored to the user’s interests.

### Dynamic Updates Based on User Input
- The system continuously refines its question-generation strategy by analyzing user inputs.
- Each response influences the RL model’s policy, ensuring questions remain relevant and engaging.
- Inputs can be text-based or speech-based (processed via Whisper).

### Multilingual Support
- Users can provide answers in various languages (e.g., English, Spanish, French, etc.).
- The system detects the input language and generates questions in the same or a user-specified language.
- Language processing is powered by the custom LLM and external NLP libraries.

### Custom LLM with Web Scraping
- We built a unique LLM by scraping diverse datasets from the web.
- The model is fine-tuned to generate creative and context-aware questions.
- It dynamically incorporates user inputs to improve its knowledge base over time.

### Speech Integration with Whisper
- OpenAI’s Whisper is used to convert text outputs (questions) into natural-sounding speech.
- Users can toggle between text-only and speech-based interactions.
- The system processes speech inputs (if provided) and converts them into text for further analysis.

## Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
