import os
from playsound import playsound
import sounddevice as sd
import queue
import json
import streamlit as st
from vosk import Model, KaldiRecognizer
import openai

# API Key
# from config import OPENAI_API_KEY
# openai.api_key = OPENAI_API_KEY

from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Vosk Model
model_path = "/Users/Administrator/T540_HHO/vosk-model-small-en-us-0.15"  # Replace with your actual model path
model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000)

# Queue for streaming audio
q = queue.Queue()

def play_audio(file_path):
    """Play a pre-recorded .wav audio file."""
    if os.path.exists(file_path):
        playsound(file_path)
    else:
        st.error(f"Audio file not found: {file_path}")

def audio_callback(indata, frames, time, status):
    """Callback to receive audio data."""
    if status:
        st.error(f"Stream Error: {status}")
    q.put(bytes(indata))

def analyze_sentiment_with_chatgpt(text):
    """Analyze sentiment and primary emotion."""
    prompt = f"""
    Analyze the sentiment of the following text based on the circumplex model of emotions. Identify the primary and secondary(if any) emotion (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation) and its intensity (Low, Medium, High):\n\n{text}
    Example:

    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]

def extract_entities_with_emotions(text):
    """Extract entities and emotions with associations."""
    prompt = f"""
    Extract the following categories of entities from the text below. For each category, list the relevant details:

    1. **People**: Names of people mentioned in the text.
    2. **Locations**: Specific locations mentioned (e.g., park, apartment, city, bookstore).
    3. **Events**: Key actions or activities described in the text (e.g., walking, kissing, watching a movie).
    4. **Environment Conditions**: The surroundings or environment (e.g., rainy, noisy, cold).
    5. **Emotions**: Identify emotions based on the circumplex model of emotions (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation) and their intensity (Low, Medium, High).
    6. **Associations**: For each emotion, provide:
       - People
       - Locations
       - Events
       - Environment Conditions

    Text:
    {text}

    Please provide the output in a structured format.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]

def process_transcription_with_chatgpt(transcription):
    """Process transcription text with sentiment analysis and entity extraction."""
    st.write("### Transcription")
    st.write(transcription)

    # Sentiment Analysis
    try:
        sentiment = analyze_sentiment_with_chatgpt(transcription)
        st.write("### Sentiment Analysis")
        st.write(sentiment)
    except Exception as e:
        st.error(f"Sentiment Analysis Error: {e}")

    # Entity Extraction with Emotions
    try:
        entities = extract_entities_with_emotions(transcription)
        st.write("### Extracted Entities and Emotions")
        st.write(entities)
    except Exception as e:
        st.error(f"Entity Extraction Error: {e}")

def continuous_transcription():
    """Continuously transcribe audio and process with ChatGPT."""
    play_audio("tell me about your day.wav")  # Play the opening message
    st.info("Listening... (say 'that's it' to end)")
    full_transcription = ""
    with sd.InputStream(samplerate=16000, channels=1, dtype="int16", callback=audio_callback):
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                st.write(f"Partial Transcript: {text}")
                full_transcription += " " + text

                if "that's it" in text.lower():
                    st.success("Stopping transcription...")
                    play_audio("thanks for sharing.wav")  # Play the closing message
                    break

    process_transcription_with_chatgpt(full_transcription)

# Streamlit App
def main():
    st.title("Neuropy HomeHub")
    st.write("Tell me about your day!")
    
    if st.button("Start Conversation"):
        try:
            continuous_transcription()
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()