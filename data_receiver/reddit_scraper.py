import requests
import pandas as pd
from datetime import datetime
import time

# 1. Налаштування пошуку
subreddit = "ukraine"
total_posts_needed = 200  # Кількість постів для повноцінного датасету
posts_data = []
after = None  # Маркер наступної сторінки Reddit
headers = {'User-Agent': 'WarWatch-Project/1.0'}

print(f"Починаємо збір повноцінного датасету з r/{subreddit}...")

# Цикл для збору декількох сторінок даних
while len(posts_data) < total_posts_needed:
    # Формуємо URL: додаємо маркер 'after' для переходу на наступну сторінку
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit=100"
    if after:
        url += f"&after={after}"

    # Виконуємо запит
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Помилка доступу: {response.status_code}")
        break

    # Парсимо JSON
    data = response.json()
    batch = data['data']['children']

    # Витягуємо потрібні поля для кожного поста
    for post in batch:
        item = post['data']
        posts_data.append({
            # Конвертуємо час Unix у формат РРРР-ММ-ДД
            'date': datetime.fromtimestamp(item['created_utc']).strftime('%Y-%m-%d'),
            'title': item['title'],  # Заголовок
            'text': item['selftext'],  # Текст поста
            'score': item['ups'],  # Рейтинг (лайки)
            'comments': item['num_comments'],  # Кількість коментарів
            'subreddit': subreddit  # Джерело
        })

    # Оновлюємо маркер для наступної сторінки
    after = data['data']['after']
    print(f"Вже зібрано: {len(posts_data)} постів...")

    # Якщо наступних сторінок немає — зупиняємось
    if not after:
        break

    # Невелика пауза, щоб Reddit не заблокував за швидкість
    time.sleep(1)

# 2. Збереження результату
# Створюємо таблицю
df = pd.DataFrame(posts_data)

# Зберігаємо файл безпосередньо у папку зі скриптом, як було раніше
filename = "reddit_dataset.csv"
df.to_csv(filename, index=False)

print("-" * 30)
print(f"Збір завершено! Файл збережено: {filename}")
print(f"Загальна кількість рядків: {len(df)}")