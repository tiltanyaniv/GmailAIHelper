# from gpt4all import GPT4All

# # Load the Llama 3 8B Instruct model
# model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")

# # Generate output
# output = model.generate("Answer this prompt by saying Hello LLM")

# # Print the output
# print(output)

import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Define the Gmail API scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    """
    Authenticate the user with Gmail API using OAuth 2.0.
    Returns a service object to interact with Gmail API.
    """
    creds = None

    # Get the path to credentials.json from the environment variable
    credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
    if not credentials_path or not os.path.exists(credentials_path):
        raise FileNotFoundError("Environment variable GOOGLE_CREDENTIALS_PATH is not set or the file does not exist.")

    # Check if token.json exists to reuse previous authentication
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    # Perform OAuth flow if there are no valid credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for future use
        with open('token.json', 'w') as token_file:
            token_file.write(creds.to_json())

    # Build and return the Gmail API service object
    return build('gmail', 'v1', credentials=creds)


if __name__ == "__main__":
    print("Connecting to Gmail API...")
    try:
        # Authenticate and build the Gmail service
        gmail_service = authenticate_gmail()

        print("Connected!")
        # List Gmail labels to confirm the connection
    except Exception as e:
        print(f"Failed to connect to Gmail API: {e}")