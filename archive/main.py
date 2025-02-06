import asyncio
import base64
import json
import os
from dotenv import load_dotenv
from hume.client import AsyncHumeClient
from hume.empathic_voice.chat.socket_client import ChatConnectOptions, ChatWebsocketConnection
from hume.empathic_voice.chat.types import SubscribeEvent
from hume.empathic_voice import UserInput, ToolCallMessage, ToolErrorMessage, ToolResponseMessage
from hume.core.api_error import ApiError
from hume import MicrophoneInterface, Stream

class WebSocketHandler:
    """Handler for containing the EVI WebSocket and associated socket handling behavior."""

    def __init__(self):
        self.socket = None
        self.byte_strs = Stream.new()

    def set_socket(self, socket: ChatWebsocketConnection):
        self.socket = socket

    async def handle_tool_call(self, message: ToolCallMessage):
        """Handles tool call messages from EVI, such as `hang_up` to stop the conversation."""
        tool_name = message.name
        tool_call_id = message.tool_call_id

        if tool_name == "hang_up":
            print("Hume AI detected a stop request. Ending conversation...")
            response = ToolResponseMessage(tool_call_id=tool_call_id, content="Conversation stopped.")
            await self.socket.send_tool_response(response)
            await self.socket.close(code=1000)  # Gracefully close WebSocket connection
            print("WebSocket connection closed due to user request.")

    async def send_hang_up(self):
        """Manually triggers Hume AI's built-in hang_up tool."""
        if self.socket is None:
            print("No active WebSocket connection.")
            return
        try:
            tool_call = ToolCallMessage(
                tool_call_id="manual_hangup",
                name="hang_up",
                parameters="{}"
            )
            await self.socket.send_tool_call(tool_call)
            print("Sent manual hang-up request to Hume AI.")
        except Exception as e:
            print(f"Error sending hang-up request: {e}")

    async def on_open(self):
        print("WebSocket connection opened.")

    async def on_message(self, message: SubscribeEvent):
        """Handles WebSocket messages, including function calls from Hume AI."""
        if message.type == "tool_call":
            if message.tool_type == "builtin":
                await self.handle_tool_call(message)
            else:
                print(f"Received custom tool call: {message.name}")
        elif message.type == "assistant_message":
            print(f"Assistant: {message.message.content}")
        elif message.type == "user_message":
            user_text = message.message.content.lower()
            print(f"User: {user_text}")
            
            # Check for stop phrases
            stop_phrases = ["stop", "cancel", "end conversation", "goodbye"]
            if any(phrase in user_text for phrase in stop_phrases):
                print("Detected stop phrase. Sending hang-up request...")
                await self.send_hang_up()
        elif message.type == "error":
            print(f"Error: {message.message}")

    async def on_close(self):
        print("WebSocket connection closed.")

    async def on_error(self, error):
        print(f"Error: {error}")

async def main():
    """Initializes the Hume AI connection and starts conversation."""
    load_dotenv()
    HUME_API_KEY = os.getenv("HUME_API_KEY")
    HUME_SECRET_KEY = os.getenv("HUME_SECRET_KEY")
    HUME_CONFIG_ID = os.getenv("HUME_CONFIG_ID")

    client = AsyncHumeClient(api_key=HUME_API_KEY)

    options = ChatConnectOptions(
        config_id=HUME_CONFIG_ID,
        secret_key=HUME_SECRET_KEY,
        builtin_tools=["hang_up"]
    )

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
                allow_user_interrupt=True,
                byte_stream=websocket_handler.byte_strs
            )
        )

        await microphone_task  # Keep the microphone running

if __name__ == "__main__":
    asyncio.run(main())
