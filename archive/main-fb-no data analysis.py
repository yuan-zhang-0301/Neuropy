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

# Load environment variables
load_dotenv()

# Firebase setup
cred = credentials.Certificate("neuropyhomehub-firebase-adminsdk-fbsvc-45769b4c9f.json")
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)
db = firestore.client()

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

def save_chat_to_firestore(chat_id, messages):
    """Save the conversation to Firestore, including only user messages with their top 3 emotions."""
    user_messages = []
    for msg in messages:
        if msg["role"] == "USER":
            # Get top 3 emotions for the message
            emotions = dict(sorted(
                msg["emotions"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]) if msg["emotions"] else {}
            
            # Create filtered message object
            filtered_msg = {
                "message": msg["message"],
                "timestamp": msg["timestamp"],
                "emotions": emotions
            }
            user_messages.append(filtered_msg)

    chat_data = {
        "chat_id": chat_id,
        "messages": user_messages,
        "timestamp": firestore.SERVER_TIMESTAMP
    }
    try:
        db.collection("Hume").document(chat_id).set(chat_data)
        print(f"Chat {chat_id} saved to Firestore (user messages with top 3 emotions).")
    except Exception as e:
        print(f"Error saving chat to Firestore: {e}")

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

        microphone_task = asyncio.create_task(
            MicrophoneInterface.start(
                socket,
                allow_user_interrupt=False,
                byte_stream=websocket_handler.byte_strs
            )
        )
        
        message_sending_task = asyncio.create_task(sending_handler(socket))
        await asyncio.gather(microphone_task, message_sending_task)

if __name__ == "__main__":
    asyncio.run(main())