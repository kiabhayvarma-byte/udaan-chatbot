import os
import io
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS
from audio_recorder_streamlit import audio_recorder

# Load environment variable agar .env me ho
load_dotenv()

st.set_page_config(
    page_title="Udaan - Ek Nayi Soch AI Chatbot",
    page_icon="🚀",
    layout="wide"
)

# Active chat models ki list (ek fail hua to script dusra try karegi)
CANDIDATE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b"
]

# Sidebar Configuration
with st.sidebar:
    st.header("🔑 API Setup")
    env_key = os.getenv("GROQ_API_KEY", "")
    api_key_input = st.text_input("Groq API Key daaliye:", value=env_key, type="password")

    st.header("🌐 Settings")
    language = st.selectbox(
        "Response Language:",
        ["Auto-Detect", "Hindi (हिन्दी)", "Bhojpuri (भोजपुरी)", "English"]
    )
    
    enable_voice_output = st.checkbox("Voice Output Enable Rakhein", value=True)
    
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
    
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# API Key Validation
if not api_key_input:
    st.warning("👈 Left side ke sidebar me apni Groq API Key daaliye chalu karne ke liye.")
    st.stop()

# Groq Client Initialization
client = Groq(api_key=api_key_input.strip())

# System Prompt
SYSTEM_PROMPT = """
You are 'Udaan - Ek Nayi Soch AI CHATBOT', an expert education consultant and teacher trainer.
You specialize in blending Digital and Traditional Pedagogies across 6 core pillars:

1. Digital and Traditional Learning (Blending chalk-and-talk with digital content, tablets, mobile tools)
2. Teaching Learning Material - TLM (Low-cost, no-cost handmade models, charts, tactile aids)
3. Low Cost Smart Class (Using basic projectors, TV screens, smartphones, offline content like DIKSHA)
4. Inclusive Education (Catering to children with special needs, slow learners, multilingual backgrounds, gender equity)
5. Activity Based Learning - ABL (Learning through play, peer learning, experiments, math labs)
6. Eco Friendly School (Green campus, seed banks, waste recycling, nature integration, kitchen gardens)

RULES:
- Provide strictly plain text or markdown conversational answers.
- NEVER output JSON schema, commentary arguments, or invoke tool calls.
- Support English, Hindi, and polite Bhojpuri naturally.
"""

st.title("🚀 Udaan - Ek Nayi Soch AI Chatbot")
st.caption("Empowering Teachers & Schools | Hindi, English & Bhojpuri")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "audio" in msg and msg["audio"]:
            st.audio(msg["audio"], format="audio/mp3")

# Voice Input Section
st.write("🎙️ **Voice Recording (Boliye):**")
audio_bytes = audio_recorder(
    text="Mic par click karein aur bolein",
    recording_color="#e8b62c",
    neutral_color="#6aa36f",
    icon_name="microphone",
    icon_size="2x"
)

user_text = ""

# Audio Transcription (Whisper)
if audio_bytes:
    with st.spinner("Aawaz transcribe ho rahi hai..."):
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "voice.wav"
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                response_format="text"
            )
            user_text = transcription
            st.info(f"🗣️ Aapne bola: {user_text}")
        except Exception as e:
            st.error(f"Voice error: {e}")

# Text Input Section
chat_input = st.chat_input("Apna sawaal likhein ya upar mic se bolein...")
if chat_input:
    user_text = chat_input

# Multi-Model Resilient Chat Function
def get_groq_response(messages_payload):
    last_error = None
    for model_name in CANDIDATE_MODELS:
        try:
            res = client.chat.completions.create(
                model=model_name,
                messages=messages_payload,
                temperature=0.7,
                max_tokens=1024,
            )
            return res.choices[0].message.content, model_name
        except Exception as err:
            last_error = err
            # Agar model decommissioned ya 404 ho, to next model try karega
            continue
    raise last_error

# Process Message
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    # Language constraint
    lang_instruction = ""
    if language == "Hindi (हिन्दी)":
        lang_instruction = "Answer strictly in Devanagari Hindi only. Plain conversation."
    elif language == "Bhojpuri (भोजपुरी)":
        lang_instruction = "Answer strictly in natural, polite Bhojpuri only. Plain conversation."
    elif language == "English":
        lang_instruction = "Answer strictly in English only. Plain conversation."

    groq_payload = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n{lang_instruction}"}
    ]
    for m in st.session_state.messages:
        groq_payload.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant"):
        with st.spinner("Jawab taiyar ho raha hai..."):
            try:
                assistant_reply, used_model = get_groq_response(groq_payload)
                st.write(assistant_reply)

                # Audio Generation (TTS)
                audio_data = None
                if enable_voice_output:
                    try:
                        tts_lang = "hi" if language in ["Hindi (हिन्दी)", "Bhojpuri (भोजपुरी)"] else "en"
                        tts = gTTS(text=assistant_reply, lang=tts_lang, slow=False)
                        audio_buffer = io.BytesIO()
                        tts.write_to_fp(audio_buffer)
                        audio_buffer.seek(0)
                        audio_data = audio_buffer.read()
                        st.audio(audio_data, format="audio/mp3")
                    except Exception:
                        pass

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_reply,
                    "audio": audio_data
                })

            except Exception as e:
                st.error(f"Groq API Error: {e}")
