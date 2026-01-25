import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Optional, List


def send_email_with_audio(
    recipient_email: str,
    subject: str,
    body_text: str,
    audio_file_path: str,
    sender_email: Optional[str] = None,
    app_password: Optional[str] = None
) -> bool:
    """
    Send an email with a WAV audio file attachment using Gmail SMTP.
    
    Args:
        recipient_email: Recipient's email address
        subject: Email subject line
        body_text: Text content for email body
        audio_file_path: Path to the WAV audio file to attach
        sender_email: Sender's Gmail address (defaults to env var GMAIL_ADDRESS)
        app_password: Gmail app password (defaults to env var GMAIL_APP_PASSWORD)
        
    Returns:
        True if email sent successfully, False otherwise
        
    Environment Variables:
        GMAIL_ADDRESS: Your Gmail email address
        GMAIL_APP_PASSWORD: Your Gmail app password (not regular password)
    """
    # Get credentials from env variables if not provided
    sender_email = sender_email or os.getenv('GMAIL_ADDRESS')
    app_password = app_password or os.getenv('GMAIL_APP_PASSWORD')
    
    # Validate inputs
    if not sender_email:
        print("[ERROR] Sender email not provided. Set GMAIL_ADDRESS env variable or pass sender_email parameter")
        return False
        
    if not app_password:
        print("[ERROR] App password not provided. Set GMAIL_APP_PASSWORD env variable or pass app_password parameter")
        return False
    
    if not recipient_email:
        print("[ERROR] Recipient email is required")
        return False
        
    # Validate audio file exists
    audio_path = Path(audio_file_path)
    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_file_path}")
        return False
        
    if not audio_path.suffix.lower() == '.wav':
        print(f"[WARNING] File extension is not .wav: {audio_file_path}")
    
    try:
        # Create message container
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Attach text body
        msg.attach(MIMEText(body_text, 'plain'))
        
        # Attach audio file
        print(f"[INFO] Attaching audio file: {audio_path.name}")
        with open(audio_file_path, 'rb') as audio_file:
            audio_data = audio_file.read()
            audio_attachment = MIMEAudio(audio_data, 'wav')
            audio_attachment.add_header(
                'Content-Disposition',
                f'attachment; filename="{audio_path.name}"'
            )
            msg.attach(audio_attachment)
        
        # Connect to Gmail SMTP server using SSL (port 465)
        print(f"[INFO] Connecting to Gmail SMTP server (SSL)...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            print(f"[INFO] Logging in as {sender_email}")
            server.login(sender_email, app_password)
            
            print(f"[INFO] Sending email to {recipient_email}")
            server.send_message(msg)
            
        print(f"[SUCCESS] Email sent successfully to {recipient_email}")
        return True
        
    except smtplib.SMTPAuthenticationError:
        print("[ERROR] Authentication failed. Check your email and app password")
        print("[INFO] Make sure you're using an App Password, not your regular Gmail password")
        print("[INFO] Generate App Password at: https://myaccount.google.com/apppasswords")
        return False
        
    except smtplib.SMTPException as e:
        print(f"[ERROR] SMTP error occurred: {e}")
        return False
        
    except FileNotFoundError:
        print(f"[ERROR] Audio file not found: {audio_file_path}")
        return False
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

def send_email_with_attachments(
    recipient_email: str,
    subject: str,
    body_text: str,
    attachment_paths: List[str] = None,
    sender_email: Optional[str] = None,
    app_password: Optional[str] = None
) -> bool:
    """
    Send an email with multiple file attachments using Gmail SMTP.
    
    Args:
        recipient_email: Recipient's email address
        subject: Email subject line
        body_text: Text content for email body
        attachment_paths: List of file paths to attach
        sender_email: Sender's Gmail address (defaults to env var GMAIL_ADDRESS)
        app_password: Gmail app password (defaults to env var GMAIL_APP_PASSWORD)
        
    Returns:
        True if email sent successfully, False otherwise
    """
    # Get credentials from env variables if not provided
    sender_email = sender_email or os.getenv('GMAIL_ADDRESS')
    app_password = app_password or os.getenv('GMAIL_APP_PASSWORD')
    
    # Validate inputs
    if not sender_email:
        print("[ERROR] Sender email not provided. Set GMAIL_ADDRESS env variable or pass sender_email parameter")
        return False
        
    if not app_password:
        print("[ERROR] App password not provided. Set GMAIL_APP_PASSWORD env variable or pass app_password parameter")
        return False
    
    if not recipient_email:
        print("[ERROR] Recipient email is required")
        return False
    
    try:
        # Create message container
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Attach text body
        msg.attach(MIMEText(body_text, 'plain'))
        
        # Attach files if provided
        if attachment_paths:
            for file_path in attachment_paths:
                if not os.path.exists(file_path):
                    print(f"[WARNING] Attachment not found, skipping: {file_path}")
                    continue
                
                file_name = os.path.basename(file_path)
                print(f"[INFO] Attaching file: {file_name}")
                
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                    
                    # Determine attachment type based on extension
                    if file_path.lower().endswith('.wav'):
                        attachment = MIMEAudio(file_data, 'wav')
                    else:
                        attachment = MIMEBase('application', 'octet-stream')
                        attachment.set_payload(file_data)
                        encoders.encode_base64(attachment)
                    
                    attachment.add_header(
                        'Content-Disposition',
                        f'attachment; filename="{file_name}"'
                    )
                    msg.attach(attachment)
        
        # Connect to Gmail SMTP server using SSL (port 465)
        print(f"[INFO] Connecting to Gmail SMTP server (SSL)...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            print(f"[INFO] Logging in as {sender_email}")
            server.login(sender_email, app_password)
            
            print(f"[INFO] Sending email to {recipient_email}")
            server.send_message(msg)
            
        print(f"[SUCCESS] Email sent successfully to {recipient_email}")
        return True
        
    except smtplib.SMTPAuthenticationError:
        print("[ERROR] Authentication failed. Check your email and app password")
        print("[INFO] Make sure you're using an App Password, not your regular Gmail password")
        print("[INFO] Generate App Password at: https://myaccount.google.com/apppasswords")
        return False
        
    except smtplib.SMTPException as e:
        print(f"[ERROR] SMTP error occurred: {e}")
        return False
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

# if __name__ == "__main__":
#     # Example usage
#     print("=" * 60)
#     print("Email Sender with Audio Attachment")
#     print("=" * 60)
#
#     # Check for environment variables
#     sender = os.getenv('GMAIL_ADDRESS')
#     password = os.getenv('GMAIL_APP_PASSWORD')
#
#     if not sender or not password:
#         print("\n[SETUP] Required environment variables:")
#         print("  export GMAIL_ADDRESS='your-email@gmail.com'")
#         print("  export GMAIL_APP_PASSWORD='your-app-password'")
#         print("\n[NOTE] Get App Password from: https://myaccount.google.com/apppasswords")
#         exit(1)
#
#     # Example configuration
#     recipient = input("\nEnter recipient email: ").strip()
#     if not recipient:
#         print("[ERROR] Recipient email is required")
#         exit(1)
#
#     subject = input("Enter email subject: ").strip() or "Audio File Attachment"
#     body = input("Enter email body text: ").strip() or "Please find the attached audio file."
#     audio_file = input("Enter path to WAV audio file: ").strip()
#
#     if not audio_file:
#         print("[ERROR] Audio file path is required")
#         exit(1)
#
#     # Send email
#     success = send_email_with_audio(
#         recipient_email=recipient,
#         subject=subject,
#         body_text=body,
#         audio_file_path=audio_file
#     )
#
#     if success:
#         print("\n✓ Email sent successfully!")
#     else:
#         print("\n✗ Failed to send email")
