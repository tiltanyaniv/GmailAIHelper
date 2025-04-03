# 📬 GmailAIHelper

GmailAIHelper is a Python program that connects to your Gmail inbox, analyzes the latest emails using an AI model (Phi-3-mini), and classifies each email into a category like **Work**, **School**, or **Shopping**. It also shows a pie chart of the results!

---

## ✅ Features

- Connects to Gmail using OAuth 2.0
- Uses a local AI model to analyze emails
- Caches AI responses in Redis
- Shows categorized email results with a pie chart

---

## 🛠️ How to Run

1. **Clone the project**  
   ```bash
   git clone https://github.com/your-username/GmailAIHelper.git
   cd GmailAIHelper

2. Install the required packages
   ```bash
   pip install -r requirements.txt
   
3. Set your Gmail credentials path
   Create a Google Cloud project with Gmail API access and download your credentials.json, then run:
   ```bash
   export GOOGLE_CREDENTIALS_PATH="/path/to/credentials.json"

4. Make sure Redis is running,
   This project uses Redis to cache responses.
   
5. Run the script
   ```bash
   python GmailAIHelper.py
   

## 📦 Example Output
   ```bash
    {
      "Category": "Work",
      "Priority": "Important",
      "RequiresResponse": "No"
    }
   ```
   
👥 Made by
Tiltan Yaniv

📜 License

This project is for educational purposes.
---
