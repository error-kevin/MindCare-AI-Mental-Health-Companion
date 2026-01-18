import streamlit as st
import joblib
import re
import string
import google.generativeai as genai 

# --- 1. CONFIGURATION (API KEY) ---
GOOGLE_API_KEY = "AIzaSyBoSWh1tvfb-B4Etp7s-AkFIloiGySt5pw"

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

active_model = None

try:
 
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
          
            if 'flash' in m.name:
                active_model = genai.GenerativeModel(m.name)
                break
    
    if not active_model:
        active_model = genai.GenerativeModel('gemini-pro')

except Exception as e:
    st.error(f"API Connection Error: {e}")

@st.cache_resource
def load_local_model():
    model = joblib.load('emotion_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

try:
    local_model, vectorizer = load_local_model()
except:
    st.error("🚨 Local model files missing. Run 'train_model.py' first.")
    st.stop()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# ---  HYBRID LOGIC ---
def get_hybrid_response(user_text):
    # Step A: Local Model Analysis
    cleaned_text = clean_text(user_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    detected_mood = local_model.predict(vectorized_text)[0]

    # Step B: Google AI Response
    prompt = f"""
    Act as a professional, empathetic mental health therapist.
    Context: User said "{user_text}". Local Model detected: {detected_mood.upper()}.
    If text implies anxiety/sadness but model says JOY, trust the text.
    Provide a short, comforting response + 1 psychological tip.
    If suicide/death mentioned, ONLY provide helpline numbers.
    """
    
    try:
      
        response = active_model.generate_content(prompt)
        return response.text, detected_mood
    except Exception as e:
        return f"⚠️ Connection Failed. Error: {str(e)}", detected_mood

# ---  UI SETUP ---
st.set_page_config(page_title="MindCare Hybrid AI", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
    .stChatInput {border-radius: 20px;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("🧠 MindCare Hybrid")
    st.success("✅ System Status: Online")
    st.caption("Architecture: Stable v1")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.title("🧠 MindCare: AI Therapist")
st.caption("Hybrid Model: Accurate Detection + Human-like Responses")

# Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if user_input := st.chat_input("How are you feeling today?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            
            if active_model:
                ai_reply, mood_label = get_hybrid_response(user_input)
            else:
                ai_reply = "⚠️ Error: No Gemini model found for this API Key."
                mood_label = "Error"
            
            st.markdown(ai_reply)
            st.caption(f"Detected Mood: {mood_label.upper()}")
    
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})