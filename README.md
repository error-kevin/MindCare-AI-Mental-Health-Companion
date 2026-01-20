# MindCare AI: Mental Health Companion Chatbot

MindCare AI is a Hybrid AI-powered chatbot designed to support student mental wellness. It detects user emotions through sentiment analysis and generates empathetic, motivational responses along with relaxation tips to help manage stress, anxiety, and loneliness.

## Key Features
- Emotion Detection: Uses a Machine Learning model (LinearSVC) to classify 6 core emotions: Joy, Sadness, Anger, Fear, Love, and Surprise.
- Empathetic Conversations: Integrated with Google Gemini Pro API to provide human-like, therapeutic dialogues.
- Safety Protocols: Automatically detects self-harm keywords and immediately provides emergency helpline information.
- Relaxation Techniques: Suggests actionable psychological exercises such as the 4-7-8 breathing technique and grounding exercises.
- Privacy First: Offers a safe, anonymous space for students to express feelings without fear of judgment.

## Tech Stack
- Language: Python
- Frontend: Streamlit (Interactive Web Interface)
- Machine Learning: Scikit-Learn (LinearSVC Algorithm)
- Generative AI: Google Gemini Pro API
- Data Processing: Pandas, NumPy, NLTK, Joblib

## Project Architecture
The project follows a Hybrid AI Pipeline:
1. Preprocessing: Cleans raw user text using Regex and NLTK.
2. Local Prediction: The trained Scikit-Learn model detects the primary emotion.
3. Generative Response: The detected emotion and user context are sent to Google Gemini Pro to generate a warm, supportive response.



## Installation and Setup

1. Clone the Repository:
   git clone [https://github.com/error-kevin/MindCare-AI-Mental-Health-Companion.git](https://github.com/error-kevin/MindCare-AI-Mental-Health-Companion)

   cd mental-health-chatbot

3. Install Dependencies:
   pip install -r requirements.txt

4. Configure API Key:
   Open app.py and enter your Google Gemini API Key in the designated section.

5. Run the Application:
   python -m streamlit run app.py

## Dataset
The local emotion detection model was trained on a comprehensive dataset of 16,000+ labeled emotional samples to ensure high classification accuracy (~88%).

## Acknowledgments
This project was developed as part of the IBM SkillsBuild Internship program in collaboration with Edunet Foundation.

---
Developed by: Keshav Joshi
