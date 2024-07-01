import requests

# Remplacez par votre token de bot
TELEGRAM_API_URL = "https://api.telegram.org/bot7010066680:AAHJxpChwtfiK0PBhJFAGCgn6sd4HVOVARI"

# Fonction pour obtenir les mises à jour du bot
def get_updates():
    response = requests.get(f"{TELEGRAM_API_URL}/getUpdates")
    if response.status_code == 200:
        updates = response.json()
        print(updates)
    else:
        print(f"Échec de l'obtention des mises à jour: {response.text}")

# Fonction pour envoyer un message Telegram
def send_telegram_message(chat_id, message):
    payload = {
        'chat_id': chat_id,
        'text': message
    }
    response = requests.post(f"{TELEGRAM_API_URL}/sendMessage", data=payload)
    if response.status_code == 200:
        print("Message envoyé avec succès")
    else:
        print(f"Échec de l'envoi du message: {response.text}")

# Tester l'obtention de mises à jour
if __name__ == "__main__":
    get_updates()






