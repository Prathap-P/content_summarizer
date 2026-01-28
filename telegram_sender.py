import os
import requests
from typing import Optional, List
from pathlib import Path


# Telegram message character limit
TELEGRAM_MAX_MESSAGE_LENGTH = 4096


def split_message(message: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> List[str]:
    """
    Split a long message into multiple chunks respecting Telegram's character limit.
    Maximizes use of character limit and splits at word boundaries.
    
    Args:
        message: The message to split
        max_length: Maximum length per message chunk (default: 4096)
        
    Returns:
        List of message chunks
    """
    if len(message) <= max_length:
        return [message]
    
    chunks = []
    remaining = message
    
    while remaining:
        # If remaining text fits in one message, add it and break
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break
        
        # Take as much as possible up to max_length
        chunk = remaining[:max_length]
        
        # Find the last space to avoid breaking words
        last_space = chunk.rfind(' ')
        last_newline = chunk.rfind('\n')
        
        # Use the last whitespace (prefer newline over space)
        split_pos = max(last_space, last_newline)
        
        if split_pos > 0:
            # Split at word boundary
            chunks.append(chunk[:split_pos])
            remaining = remaining[split_pos:].lstrip()
        else:
            # No space found, force split at max_length
            chunks.append(chunk)
            remaining = remaining[max_length:]
    
    return chunks


def send_telegram_with_audio(
    chat_id: str,
    message: str,
    audio_file_path: str,
    bot_token: Optional[str] = None
) -> bool:
    """
    Send a Telegram message with a single audio file attachment.
    
    Args:
        chat_id: Telegram chat ID or username (@username)
        message: Text message to send
        audio_file_path: Path to the audio file to attach
        bot_token: Telegram bot token (defaults to env var TELEGRAM_BOT_TOKEN)
        
    Returns:
        True if message sent successfully, False otherwise
        
    Environment Variables:
        TELEGRAM_BOT_TOKEN: Your Telegram bot token from @BotFather
    """
    # Get bot token from env variable if not provided
    bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
    
    # Validate inputs
    if not bot_token:
        print("[ERROR] Bot token not provided. Set TELEGRAM_BOT_TOKEN env variable or pass bot_token parameter")
        return False
    
    if not chat_id:
        print("[ERROR] Chat ID is required")
        return False
    
    # Validate audio file exists
    audio_path = Path(audio_file_path)
    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_file_path}")
        return False
    
    try:
        # Split message if it exceeds character limit
        message_chunks = split_message(message)
        
        print(f"[DEBUG] Message length: {len(message)} chars")
        print(f"[DEBUG] Split into {len(message_chunks)} chunks")
        for idx, chunk in enumerate(message_chunks, 1):
            print(f"[DEBUG] Chunk {idx} length: {len(chunk)} chars")
        
        print(f"[INFO] Sending message to Telegram chat {chat_id} ({len(message_chunks)} part(s))...")
        text_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        # Send all message chunks in order
        for i, chunk in enumerate(message_chunks, 1):
            print(f"[INFO] Sending message part {i}/{len(message_chunks)}...")
            text_response = requests.post(text_url, data={
                'chat_id': chat_id,
                'text': chunk
            })
            
            if not text_response.ok:
                print(f"[ERROR] Failed to send message part {i}: {text_response.text}")
                return False
        
        print(f"[SUCCESS] All message parts sent ({len(message_chunks)} message(s))")
        
        # Send audio file
        print(f"[INFO] Sending audio file: {audio_path.name}")
        audio_url = f"https://api.telegram.org/bot{bot_token}/sendAudio"
        
        with open(audio_file_path, 'rb') as audio_file:
            files = {'audio': (audio_path.name, audio_file, 'audio/wav')}
            data = {'chat_id': chat_id}
            
            audio_response = requests.post(audio_url, data=data, files=files)
            
            if not audio_response.ok:
                print(f"[ERROR] Failed to send audio: {audio_response.text}")
                return False
        
        print(f"[SUCCESS] Audio file sent successfully")
        
        # Send separator messages to denote end of transaction
        print(f"[INFO] Sending separator messages...")
        text_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        separator_messages = [
            "─" * 30,
            "─" * 30,
            "─" * 30
        ]
        
        for separator in separator_messages:
            requests.post(text_url, data={
                'chat_id': chat_id,
                'text': separator
            })
        
        print(f"[SUCCESS] Transaction completed with separators")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network error occurred: {e}")
        return False
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False


def send_telegram_with_attachments(
    chat_id: str,
    message: str,
    attachment_paths: List[str] = None,
    bot_token: Optional[str] = None
) -> bool:
    """
    Send a Telegram message with multiple file attachments.
    
    Args:
        chat_id: Telegram chat ID or username (@username)
        message: Text message to send
        attachment_paths: List of file paths to attach
        bot_token: Telegram bot token (defaults to env var TELEGRAM_BOT_TOKEN)
        
    Returns:
        True if message sent successfully, False otherwise
    """
    # Get bot token from env variable if not provided
    bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
    
    # Validate inputs
    if not bot_token:
        print("[ERROR] Bot token not provided. Set TELEGRAM_BOT_TOKEN env variable or pass bot_token parameter")
        return False
    
    if not chat_id:
        print("[ERROR] Chat ID is required")
        return False
    
    try:
        # Split message if it exceeds character limit
        message_chunks = split_message(message)
        
        print(f"[INFO] Sending message to Telegram chat {chat_id} ({len(message_chunks)} part(s))...")
        text_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        # Send all message chunks in order
        for i, chunk in enumerate(message_chunks, 1):
            print(f"[INFO] Sending message part {i}/{len(message_chunks)}...")
            text_response = requests.post(text_url, data={
                'chat_id': chat_id,
                'text': chunk
            })
            
            if not text_response.ok:
                print(f"[ERROR] Failed to send message part {i}: {text_response.text}")
                return False
        
        print(f"[SUCCESS] All message parts sent ({len(message_chunks)} message(s))")
        
        # Send attachments if provided
        if attachment_paths:
            for file_path in attachment_paths:
                if not os.path.exists(file_path):
                    print(f"[WARNING] Attachment not found, skipping: {file_path}")
                    continue
                
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                
                # Check file size (Telegram bot limit is 50MB)
                if file_size > 50 * 1024 * 1024:
                    print(f"[WARNING] File too large (>50MB), skipping: {file_name}")
                    continue
                
                print(f"[INFO] Sending file: {file_name} ({file_size / 1024:.2f} KB)")
                
                # Determine endpoint based on file type
                file_ext = file_path.lower()
                if file_ext.endswith(('.wav', '.mp3', '.ogg', '.m4a')):
                    endpoint = 'sendAudio'
                    mime_type = 'audio/wav' if file_ext.endswith('.wav') else 'audio/mpeg'
                    file_key = 'audio'
                elif file_ext.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    endpoint = 'sendPhoto'
                    mime_type = 'image/jpeg'
                    file_key = 'photo'
                else:
                    endpoint = 'sendDocument'
                    mime_type = 'application/octet-stream'
                    file_key = 'document'
                
                url = f"https://api.telegram.org/bot{bot_token}/{endpoint}"
                
                with open(file_path, 'rb') as f:
                    files = {file_key: (file_name, f, mime_type)}
                    data = {'chat_id': chat_id}
                    
                    response = requests.post(url, data=data, files=files)
                    
                    if not response.ok:
                        print(f"[ERROR] Failed to send {file_name}: {response.text}")
                        continue
                
                print(f"[SUCCESS] Sent {file_name}")
        
        print(f"[SUCCESS] All messages and attachments sent successfully")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network error occurred: {e}")
        return False
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Telegram Sender with Attachments")
    print("=" * 60)
    
    # Check for environment variables
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("\n[SETUP] Required environment variables:")
        print("  export TELEGRAM_BOT_TOKEN='your-bot-token'")
        print("  export TELEGRAM_CHAT_ID='your-chat-id'")
        print("\n[NOTE] Get bot token from: @BotFather on Telegram")
        print("[NOTE] Get chat ID from: @userinfobot on Telegram")
        exit(1)
    
    # Example configuration
    message = input("\nEnter message to send: ").strip() or "Test message from Content Analyzer"
    audio_file = input("Enter path to audio file (or press Enter to skip): ").strip()
    
    if audio_file:
        # Send with audio
        success = send_telegram_with_audio(
            chat_id=chat_id,
            message=message,
            audio_file_path=audio_file
        )
    else:
        # Send text only
        success = send_telegram_with_attachments(
            chat_id=chat_id,
            message=message
        )
    
    if success:
        print("\n✓ Message sent successfully!")
    else:
        print("\n✗ Failed to send message")
