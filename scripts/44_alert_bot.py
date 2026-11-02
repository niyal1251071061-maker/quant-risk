import os

import requests


def send_discord_alert(message):
    webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    if not webhook_url:
        return False
    response = requests.post(
        webhook_url,
        json={'content': message},
        timeout=10,
    )
    response.raise_for_status()
    return True
