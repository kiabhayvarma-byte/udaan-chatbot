import os
import io
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS
from audio_recorder_streamlit import audio_recorder

load_dotenv()

st.set_page_config(
    page_title="Udaan - Ek Nayi Soch AI Chatbot",
    page_icon="🚀",
    layout="wide"
)

# API Key priority: Streamlit secrets -> .env -> manual sidebar
api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

with st.sidebar:
    st.header("⚙️ Settings")
    if not api_key:
        api_key = st.text_input("Enter Groq API Key:", type="password")

    language = st.selectbox(
        "Response Language (Bhasa):",
        ["Hindi (हिन्दी)", "Bhojpuri (भोजपुरी)", "English (Indian Accent)", "Auto-Detect"]
    )
    
    enable_voice = st.checkbox("Voice Output On", value=True)
    voice_speed = st.checkbox("Fast Speech Mode (Tez aawaz)", value=False)
    
    st.markdown("---")
    st.markdown("### 📚 6 Pillars of Udaan:")
    st.markdown("""
    1. **Digital & Traditional Learning**
    2. **Teaching Learning Material (TLM)**
    3. **Low-Cost Smart Class**
    4. **Inclusive Education**
    5. **Activity Based Learning (ABL)**
    6. **Eco-Friendly School**
    """)
    
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if not api_key:
    st.warning("👈 Kripya sidebar mein Groq API Key darj karein.")
    st.stop()

client = Groq(api_key=api_key.strip())

SYSTEM_PROMPT = """
You are 'Udaan - Ek Nayi Soch AI CHATBOT', an energetic, practical education mentor and teacher trainer.
You specialize in 6 core pillars: Digital/Traditional Learning, TLM, Low-cost Smart Class, Inclusive Education, Activity Based Learning (ABL), and Eco-friendly Schools.

Tone & Style Rules:
- Keep responses crisp, practical, and directly actionable (avoid unnecessary long essays so voice output stays fast).
- Never output JSON, code blocks, or tool calls unless explicitly asked.
- For Hindi: Use clean, natural standard conversational Hindi.
- For Bhojpuri: Use polite, natural Bhojpuri (e.g., 'रउआ सब के प्रणाम', 'ई तरीका बहुते कारगर बा').
- For English: Use natural Indian-friendly English phrasing.
"""

st.title("🚀 Udaan - Ek Nayi Soch AI Chatbot")
st.caption("Clean Indian Voice Assistant | Hindi, English & Bhojpuri")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "audio" in msg and msg["audio"]:
            st.audio(msg["audio"], format="audio/mp3")

# Voice Input (Groq Whisper)
st.write("🎙️ **Mic se bolein:**")
audio_bytes = audio_recorder(
    text="Boliye (Click to speak)",
    recording_color="#e8b62c",
    neutral_color="#6aa36f",
    icon_name="microphone",
    icon_size="2x"
)

user_text = ""

if audio_bytes:
    with st.spinner("Sun raha hoon..."):
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                response_format="text"
            )
            user_text = transcription
            st.info(f"🗣️ Aapne bola: {user_text}")
        except Exception as e:
            st.error(f"Voice input error: {e}")

chat_input = st.chat_input("Apna sawaal likhein ya mic se bolein...")
if chat_input:
    user_text = chat_input

# Response Generation
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    lang_instruction = ""
    if language == "Hindi (हिन्दी)":
        lang_instruction = "Respond concisely in standard Hindi."
    elif language == "Bhojpuri (भोजपुरी)":
        lang_instruction = "Respond concisely in natural, respectful Bhojpuri."
    elif language == "English (Indian Accent)":
        lang_instruction = "Respond concisely in clear Indian English."

    groq_payload = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n{lang_instruction}"}
    ]
    for m in st.session_state.messages:
        groq_payload.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant"):
        with st.spinner("Jawab taiyar ho raha hai..."):
            try:
                # Ultra-fast model inference
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=groq_payload,
                    temperature=0.6,
                    max_tokens=600
                )
                assistant_reply = response.choices[0].message.content
                st.write(assistant_reply)

                # Crisp Indian Voice Engine (gTTS with Indian domain routing)
                audio_data = None
                if enable_voice:
                    try:
                        clean_text = assistant_reply.replace("*", "").replace("#", "").strip()
                        
                        # Audio settings for clean Indian accent
                        if language == "English (Indian Accent)":
                            tts = gTTS(text=clean_text, lang='en', tld='co.in', slow=voice_speed)
                        else:
                            tts = gTTS(text=clean_text, lang='hi', slow=voice_speed)

                        audio_buffer = io.BytesIO()
                        tts.write_to_fp(audio_buffer)
                        audio_buffer.seek(0)
                        audio_data = audio_buffer.read()
                        
                        # Auto-play audio response
                        st.audio(audio_data, format="audio/mp3", autoplay=True)
                    except Exception:
                        pass

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_reply,
                    "audio": audio_data
                })

            except Exception as e:
                st.error(f"Groq API Error: {e}")
