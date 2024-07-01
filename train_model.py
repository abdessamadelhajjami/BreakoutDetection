import requests


# Telegram bot configuration
TELEGRAM_API_URL = "https://api.telegram.org/bot7010066680:AAHJxpChwtfiK0PBhJFAGCgn6sd4HVOVARI/sendMessage"
TELEGRAM_CHAT_ID = "https://t.me/Breakout_Channel" 

# Fonction pour envoyer un message Telegram
def send_telegram_message(message):
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message
    }
    response = requests.post(TELEGRAM_API_URL, data=payload)
    if response.status_code == 200:
        print("Message envoyé avec succès")
    else:
        print(f"Échec de l'envoi du message: {response.text}")

# Tester l'envoi de message
if __name__ == "__main__":
    send_telegram_message("Test: Ceci est un message de test pour vérifier que le bot fonctionne correctement.")








