import asyncio
import base64
import datetime
import os
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
from hume.client import AsyncHumeClient
from hume.empathic_voice.chat.socket_client import ChatConnectOptions, ChatWebsocketConnection
from hume.empathic_voice.chat.types import SubscribeEvent
from hume.empathic_voice.types import UserInput
from hume.core.api_error import ApiError
from hume import MicrophoneInterface, Stream
import websockets
import openai

# Load environment variables
load_dotenv()

# Firebase setup
cred = credentials.Certificate("neuropyhomehub-firebase-adminsdk-fbsvc-45769b4c9f.json")
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class WebSocketHandler:
    """Handler for containing the EVI WebSocket and associated socket handling behavior."""

    def __init__(self):
        """Initialize WebSocketHandler."""
        self.socket = None
        self.byte_strs = Stream.new()
        self.chat_id = None
        self.messages = []

    def set_socket(self, socket: ChatWebsocketConnection):
        """Set the socket."""
        self.socket = socket

    async def on_open(self):
        """WebSocket connection opened."""
        print("WebSocket connection opened.")

    async def on_message(self, message: SubscribeEvent):
        """Handle a WebSocket message event."""
        scores = {}
        if message.type == "chat_metadata":
            self.chat_id = message.chat_id
            text = f"<CHAT_METADATA> Chat ID: {self.chat_id}"
        elif message.type in ["user_message", "assistant_message"]:
            role = message.message.role.upper()
            message_text = message.message.content
            text = f"{role}: {message_text}"

            # Extract emotion scores if available
            if message.from_text is False:
                scores = dict(message.models.prosody.scores)

            # Store message in memory
            self.messages.append({
                "role": role,
                "message": message_text,
                "timestamp": datetime.datetime.utcnow(),
                "emotions": scores
            })
        elif message.type == "audio_output":
            message_str: str = message.data
            message_bytes = base64.b64decode(message_str.encode("utf-8"))
            await self.byte_strs.put(message_bytes)
            return
        elif message.type == "error":
            raise ApiError(f"Error ({message.code}): {message.message}")
        else:
            text = f"<{message.type.upper()}>"

        self._print_prompt(text)

        # Print and store emotions
        if scores:
            top_3_emotions = self._extract_top_n_emotions(scores, 3)
            self._print_emotion_scores(top_3_emotions)
            print("")

    async def on_close(self):
        """WebSocket connection closed. Save chat to Firestore."""
        print("WebSocket connection closed.")
        if self.chat_id and self.messages:
            save_chat_to_firestore(self.chat_id, self.messages)

    async def on_error(self, error):
        """Handle WebSocket errors."""
        print(f"Error: {error}")

    def _print_prompt(self, text: str) -> None:
        """Print a formatted message with a timestamp."""
        now_str = datetime.datetime.utcnow().strftime("%H:%M:%S")
        print(f"[{now_str}] {text}")

    def _extract_top_n_emotions(self, emotion_scores: dict, n: int) -> dict:
        """Extract top N emotions based on confidence scores."""
        sorted_emotions = sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)
        return {emotion: score for emotion, score in sorted_emotions[:n]}

    def _print_emotion_scores(self, emotion_scores: dict) -> None:
        """Print emotions and scores."""
        formatted_emotions = ' | '.join([f"{emotion} ({score:.2f})" for emotion, score in emotion_scores.items()])
        print(f"|{formatted_emotions}|")

def analyze_hume_transcript(transcript, top_emotions):
    """Analyze the transcript from Hume conversations using OpenAI API."""
    print("\n--- Full Transcript from Hume ---")
    print(transcript)
    print("\n--- Top Emotions Detected by Hume ---")
    for emotion, score in top_emotions.items():
        print(f"{emotion}: {score:.2f}")
    
    try:
        # Analyze sentiment
        sentiment = analyze_sentiment_with_chatgpt(transcript, top_emotions)
        print("\n--- Sentiment Analysis ---")
        print(sentiment)
    except Exception as e:
        print(f"Sentiment Analysis Error: {e}")
        sentiment = ""

    try:
        # Extract entities and emotions
        entities = extract_entities_with_emotions(transcript, top_emotions)
        print("\n--- Extracted Entities and Emotions ---")
        print(entities)
    except Exception as e:
        print(f"Entity Extraction Error: {e}")
        entities = ""

    return sentiment, entities

def save_chat_to_firestore(chat_id, messages):
    """Save the conversation to Firestore with structured analysis."""
    try:
        if not chat_id:
            raise ValueError("chat_id is required")
            
        # Combine all user messages into one transcript
        transcript = " ".join([
            msg["message"] 
            for msg in messages 
            if msg["role"] == "USER"
        ])
        
        if not transcript:
            print("Warning: No user messages found to save")
            return
            
        # Aggregate emotions across all user messages
        all_emotions = {}
        for msg in messages:
            if msg["role"] == "USER" and msg["emotions"]:
                for emotion, score in msg["emotions"].items():
                    all_emotions[emotion] = max(score, all_emotions.get(emotion, 0))
        
        # Get top 3 emotions overall
        top_emotions = dict(sorted(
            all_emotions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3])
        
        # Get OpenAI analysis
        emotional_analysis, empathetic_feedback = analyze_sentiment_with_chatgpt(transcript, top_emotions)
        associations = extract_entities_with_emotions(transcript, top_emotions)

        # Structure the data for Firebase
        chat_data = {
            "transcript": transcript,
            "top_emotions": {
                emotion: f"{score:.2f}" for emotion, score in top_emotions.items()
            },
            "emotional_analysis": emotional_analysis,
            "empathetic_feedback": empathetic_feedback,
            "emotional_associations": associations,
            "timestamp": firestore.SERVER_TIMESTAMP
        }
        
        # Save to Firebase
        db.collection("Hume").document(chat_id).set(chat_data)
        
        # Print the structured analysis
        print("\n=== Analysis Results ===")
        print("\nTranscript:")
        print(transcript)
        print("\nTop Emotions:")
        for emotion, score in top_emotions.items():
            print(f"{emotion}: {score:.2f}")
        print("\nEmotional Analysis:")
        print(emotional_analysis)
        print("\nEmpathetic Feedback:")
        print(empathetic_feedback)
        print("\nEmotional Associations:")
        print(associations)
        
    except Exception as e:
        print(f"Error saving chat to Firestore: {e}")
        print(f"Error type: {type(e).__name__}")
        raise

async def sending_handler(socket: ChatWebsocketConnection):
    """Send a message over the WebSocket."""
    await asyncio.sleep(3)
    user_input_message = UserInput(text="Hello there!")
    await socket.send_user_input(user_input_message)

async def main() -> None:
    load_dotenv()

    HUME_API_KEY = os.getenv("HUME_API_KEY")
    HUME_SECRET_KEY = os.getenv("HUME_SECRET_KEY")
    HUME_CONFIG_ID = os.getenv("HUME_CONFIG_ID")

    client = AsyncHumeClient(api_key=HUME_API_KEY)
    options = ChatConnectOptions(config_id=HUME_CONFIG_ID, secret_key=HUME_SECRET_KEY)

    websocket_handler = WebSocketHandler()

    async with client.empathic_voice.chat.connect_with_callbacks(
        options=options,
        on_open=websocket_handler.on_open,
        on_message=websocket_handler.on_message,
        on_close=websocket_handler.on_close,
        on_error=websocket_handler.on_error
    ) as socket:
        websocket_handler.set_socket(socket)
        
        try:
            microphone_task = asyncio.create_task(
                MicrophoneInterface.start(
                    socket,
                    allow_user_interrupt=False,
                    byte_stream=websocket_handler.byte_strs
                )
            )
            
            message_sending_task = asyncio.create_task(sending_handler(socket))
            await asyncio.gather(microphone_task, message_sending_task)
        except websockets.exceptions.ConnectionClosedOK:
            print("Chat session ended normally.")
        except Exception as e:
            print(f"An error occurred: {e}")

def analyze_sentiment_with_chatgpt(text, top_emotions):
    """Analyze sentiment using ChatGPT, incorporating Hume's emotion analysis."""
    emotions_str = ", ".join([f"{emotion} (probability: {score:.2f})" for emotion, score in top_emotions.items()])
    
    prompt = f"""Analyze the emotional state and sentiment of this person in two distinct parts.
Hume's voice analysis detected these top emotions: {emotions_str}

Part 1 - Emotional Analysis (max 5 sentences):
Start with "Hey there, it sounds like you're feeling..." and describe:
- Their current emotional state
- Direct causes of these emotions
- Specific situations they mentioned
- Keep it purely descriptive without interpretation

Part 2 - Empathetic Response (2-3 sentences):
- Provide brief validation
- Offer gentle support
- Keep it concise and warm

Text:
{text}

Return the two parts separated by |||"""

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a concise, empathetic counselor who speaks directly to people about their emotions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0
    )
    analysis, feedback = response.choices[0].message.content.split("|||")
    return analysis.strip(), feedback.strip()

def extract_entities_with_emotions(text, top_emotions):
    """Extract entities and emotions using ChatGPT."""
    emotions_str = ", ".join([f"{emotion} (probability: {score:.2f})" for emotion, score in top_emotions.items()])
    
    prompt = f"""Analyze the associations for each detected emotion in the voice analysis:
{emotions_str}

For each emotion, provide:
1. Associated People: Who was mentioned
2. Associated Locations: Where this emotion was expressed
3. Associated Events: What triggered this emotion
4. Associated Environment: Surrounding conditions
5. Expressed Intensity: Rate as Low/Medium/High based on their expression, NOT the probability score

Format as a clear list for each emotion.

Text:
{text}"""

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who provides structured emotional analysis."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    asyncio.run(main())