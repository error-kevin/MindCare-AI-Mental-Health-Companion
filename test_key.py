# test_key.py
import google.generativeai as genai

KEY = "AIzaSyBoSWh1tvfb-B4Etp7s-AkFIloiGySt5pw" 

try:
    genai.configure(api_key=KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Hello")
    print("✅ SUCCESS! Key is working.")
    print("AI Response:", response.text)
except Exception as e:
    print("❌ ERROR! Key kaam nahi kar rahi.")
    print("Reason:", e)