from gpt4all import GPT4All
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Load the Llama 3 8B Instruct model
model = GPT4All("Phi-3-mini-4k-instruct.Q4_0.gguf")


# Model configuration settings
# MODEL_SETTINGS = {
#     "temp": 0.0,  # Controls randomness; lower = more focused and deterministic
#     "top_p": 0.0,       # Nucleus sampling; lower = more predictable
#     "top_k": 1,        # Limits token selection pool; smaller = more focused
#     "repeat_penalty": .0,  # Penalizes repetitive sequences; >1 reduces repetitions
#     "max_tokens": 150,

# }

MODEL_SETTINGS = {
    "temp": 0.0,            # Strictly deterministic output
    "top_p": 0.0,           # Disable nucleus sampling for predictability
    "top_k": 1,             # Always pick the most probable token
    "repeat_penalty": 1.18, # Penalize repetition
    "repeat_last_n": 64,    # Apply penalty to the last 64 tokens
    "max_tokens": 35,      # Limit the response length
    "n_batch": 8,           # Token processing batch size
    "n_predict": None,      # Use max_tokens if unset
    "streaming": False,     # Generate the complete JSON object before returning
}
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

# Updated process_emails_with_llm function
def process_emails_with_llm(service):
    """
    Fetch the latest email from the inbox and print its subject and sender.
    """
    try:
        # Get the list of messages in the inbox
        results = service.users().messages().list(userId='me', maxResults=10).execute()
        messages = results.get('messages', [])

        if not messages:
            print("No messages found.")
            return

        print("Last 100 Emails:")
        # Get the details of the latest message
        for message_metadata in messages:
            # Fetch the email details for each message
            message_id = message_metadata['id']
            message = service.users().messages().get(userId='me', id=message_id, format='metadata').execute()

            # Extract headers from the message
            headers = message.get('payload', {}).get('headers', [])
            subject = next((header['value'] for header in headers if header['name'] == 'Subject'), "No Subject")
            sender = next((header['value'] for header in headers if header['name'] == 'From'), "Unknown Sender")

            prompt = (
                f"<user>\n"
                f"You are an AI system that only outputs JSON objects. Your task is to classify the following email into a predefined JSON format.\n\n"
                f"Email details:\n"
                f"Subject: \"{subject}\"\n"
                f"Sender: \"{sender}\"\n\n"
                f"### Output Format\n"
                f"Respond only in this JSON format:\n"
                f"{{\n"
                f"    \"Category\": \"<Work, School, or Shopping>\",\n"
                f"    \"Priority\": \"<Important or Normal>\",\n"
                f"    \"RequiresResponse\": \"<Yes or No>\"\n"
                f"}}\n\n"
                f"### Example Response\n"
                f"{{\n"
                f"   \"Category\": \"Work\",\n"
                f"   \"Priority\": \"Normal\",\n"
                f"   \"RequiresResponse\": \"No\"\n"
                f"}}\n\n"
                f"### Important Instructions\n"
                f"1. Output *only* a single JSON object in the exact format provided above.\n"
                f"2. *Do not include any additional text, comments, explanations, or blank lines.*\n"
                f"3. Your response *must* be a valid JSON object with proper capitalization, syntax (double quotes, etc.), and no trailing commas.\n"
                f"4. If you cannot classify the email, leave the values blank (e.g., \"Category\": \"\") but maintain the JSON structure.\n\n"
                f"Now classify the email:\n"
                f"%1<end>\n"
                f"<assistant>\n"
                f"%2<end>\n"
            )

            # Generate LLM response
            llm_response = model.generate(prompt, **MODEL_SETTINGS)

            print(f"LLM Response: {llm_response}\n")

    except Exception as error:
        print(f"An error occurred while fetching the email: {error}")


if __name__ == "__main__":
    print("Connecting to Gmail API...")
    try:
        # Authenticate and build the Gmail service
        gmail_service = authenticate_gmail()

        print("Connected!")
        # Get the latest email and print its subject and sender
        process_emails_with_llm(gmail_service)
    except Exception as e:
        print(f"Failed to connect to Gmail API: {e}")