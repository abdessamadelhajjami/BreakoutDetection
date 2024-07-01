import requests

# Remplacez par votre token de bot
TELEGRAM_API_URL = "https://api.telegram.org/bot7010066680:AAHJxpChwtfiK0PBhJFAGCgn6sd4HVOVARI"

# Fonction pour obtenir les mises à jour du bot
def get_updates():
    response = requests.get(f"{TELEGRAM_API_URL}/getUpdates")
    if response.status_code == 200:
        updates = response.json()
        print(updates)
        return updates
    else:
        print(f"Échec de l'obtention des mises à jour: {response.text}")
        return None

# Tester l'obtention de mises à jour
if __name__ == "__main__":
    updates = get_updates()
    if updates and "result" in updates and len(updates["result"]) > 0:
        for update in updates["result"]:
            print(update)
    else:
        print("Aucune mise à jour trouvée.")




