import os
import sounddevice as sd
import soundfile as sf
import tempfile
import openai
import simpleaudio as sa
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
import time

# Load environment variables (including OPENAI_API_KEY)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Firebase Admin SDK
cred = credentials.Certificate("neuropyhomehub-firebase-adminsdk-fbsvc-45769b4c9f.json")
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# ------------------------------------------------
def save_to_firestore(transcription, sentiment, entities):
    """
    Save the transcription and analysis results to Firestore.
    Each conversation is stored as a document in the "conversations" collection.
    """
    doc_ref = db.collection("conversations").document()  # Auto-generated ID
    data = {
        "transcription": transcription,
        "sentiment": sentiment,
        "entities": entities,
        "timestamp": firestore.SERVER_TIMESTAMP
    }
    try:
        doc_ref.set(data)
        print("Data saved to Firestore!")
    except Exception as e:
        print(f"Error saving to Firestore: {e}")

# ------------------------------------------------
def play_audio(file_path):
    """Play a pre-recorded .wav audio file."""
    if os.path.exists(file_path):
        try:
            wave_obj = sa.WaveObject.from_wave_file(file_path)
            play_obj = wave_obj.play()
            play_obj.wait_done()
        except Exception as e:
            print(f"Error playing audio: {e}")
    else:
        print(f"Audio file not found: {file_path}")

# ------------------------------------------------
def record_audio(filename, duration=5, fs=16000):
    """Record audio for a short duration and save to a WAV file."""
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()  # Wait until recording is finished
    sf.write(filename, recording, fs)
    print("Recording complete.")
    return filename

# ------------------------------------------------
def transcribe_with_whisper(filename):
    """Transcribe a recorded audio file using the Whisper API."""
    print("Transcribing audio with Whisper API...")
    with open(filename, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]

# ------------------------------------------------
def analyze_sentiment_with_chatgpt(text):
    prompt = f"""Analyze the sentiment of the following text based on the circumplex model of emotions.
Identify the primary and secondary (if any) emotion (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation) and its intensity (Low, Medium, High):

{text}"""
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

# ------------------------------------------------
def extract_entities_with_emotions(text):
    prompt = f"""Extract the following categories of entities from the text below. For each category, list the relevant details:

1. People: Names of people mentioned.
2. Locations: Specific locations (e.g., park, apartment, city, bookstore).
3. Events: Key actions or activities described.
4. Environment Conditions: The surroundings or environment.
5. Emotions: Identify emotions based on the circumplex model (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation) and their intensity (Low, Medium, High).
6. Associations: For each emotion, provide People, Locations, Events, and Environment Conditions.

Text:
{text}

Please provide the output in a structured format."""
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

# ------------------------------------------------
def process_transcription_with_chatgpt(transcription):
    print("\n--- Full Transcription ---")
    print(transcription)
    sentiment = ""
    entities = ""
    try:
        sentiment = analyze_sentiment_with_chatgpt(transcription)
        print("\n--- Sentiment Analysis ---")
        print(sentiment)
    except Exception as e:
        print(f"Sentiment Analysis Error: {e}")
    try:
        entities = extract_entities_with_emotions(transcription)
        print("\n--- Extracted Entities and Emotions ---")
        print(entities)
    except Exception as e:
        print(f"Entity Extraction Error: {e}")
    save_to_firestore(transcription, sentiment, entities)

# ------------------------------------------------
def continuous_transcription():
    play_audio("tell me about your day.wav")
    print("Listening... (say 'that's it' in your speech to end, or type 'stop' after a chunk to end)")
    full_transcription = ""
    while True:
        print("\n---- New Chunk ----")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_audio_file = tmp.name
        record_audio(temp_audio_file, duration=5)
        transcript = transcribe_with_whisper(temp_audio_file)
        print("Partial Transcript: " + transcript)
        full_transcription += " " + transcript
        # Check if the termination phrase is in the transcript.
        if "that's it" in transcript.lower():
            print("Termination phrase detected in speech. Stopping recording...")
            play_audio("thanks for sharing.wav")
            break
        # Check for user command to stop.
        user_input = input("Press Enter to continue recording, or type 'stop' to end: ").strip().lower()
        if user_input == "stop":
            print("Stop command received. Stopping recording...")
            play_audio("thanks for sharing.wav")
            break
    process_transcription_with_chatgpt(full_transcription)

# ------------------------------------------------
def main():
    print("Neuropy HomeHub")
    print("Tell me about your day!")
    user_input = input("Type 'start' to start conversation, or 'quit' to exit: ").strip().lower()
    if user_input == "start":
        continuous_transcription()
    else:
        print("Exiting.")

if __name__ == "__main__":
    main()