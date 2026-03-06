import os
import json
import asyncio
from telethon import TelegramClient
from dotenv import load_dotenv
from datetime import datetime

# Завантажуємо налаштування
load_dotenv()
api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")


async def scrape_telegram():
    # Створюємо клієнта (файл 'anon.session' з'явиться у папці)
    async with TelegramClient('anon', api_id, api_hash) as client:
        entity = 'DeepStateUA'  # Канал для моніторингу
        print(f"Розпочинаємо збір даних з @{entity}...")

        messages = []
        # Збираємо останні 15 повідомлень
        async for message in client.iter_messages(entity, limit=15):
            if message.text:
                messages.append({
                    "source": f"telegram/@{entity}",
                    "text": message.text[:1000],
                    "date": message.date.isoformat(),
                    "collected_at": datetime.now().isoformat()
                })

        # Збереження у папку data/raw згідно з правилами проєкту
        output_dir = "data/raw"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "telegram_data.json")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=4)

        print(f"Готово! Зібрано {len(messages)} повідомлень. Файл: {file_path}")


if __name__ == "__main__":
    asyncio.run(scrape_telegram())