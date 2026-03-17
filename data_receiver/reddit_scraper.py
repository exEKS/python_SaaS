import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone

# 1. Налаштування
subreddits = ["CredibleDefense"]
# Встановлюємо ціль: 3 роки тому
stop_date = datetime.now(timezone.utc) - timedelta(days=3 * 365)
headers = {'User-Agent': 'WarWatch-Research-Student/2.0'}

all_posts = []

print(f" Починаємо збір даних з тредів: {', '.join(subreddits)}...")

# Цикл по кожному сабреддіту
for subreddit in subreddits:
    print(f"\n--- Скануємо r/{subreddit} ---")
    after = None  # Маркер для переходу на наступну сторінку

    while True:
        # Формуємо запит (максимум 100 постів за раз)
        url = f"https://www.reddit.com/r/{subreddit}/new.json?limit=100"
        if after:
            url += f"&after={after}"

        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f" Помилка на r/{subreddit}: {response.status_code}. Переходимо до наступного.")
            break

        data = response.json()
        posts = data['data']['children']

        if not posts:
            print(f" У r/{subreddit} більше постів немає.")
            break

        last_post_date = None

        for post in posts:
            item = post['data']
            # Конвертуємо час у формат UTC для порівняння
            post_date = datetime.fromtimestamp(item['created_utc'], tz=timezone.utc)
            last_post_date = post_date

            # Перевіряємо, чи ми не зайшли далі, ніж за 3 роки
            if post_date < stop_date:
                break

            all_posts.append({
                'subreddit': subreddit, # Додаємо назву треду для датасету
                'date': post_date.strftime('%Y-%m-%d %H:%M:%S'),
                'title': item['title'],
                'text': item['selftext'],
                'score': item['ups'],
                'comments': item['num_comments']
            })

        print(f" Зібрано пакет. Всього вже: {len(all_posts)} постів. Остання дата: {last_post_date.strftime('%Y-%m-%d')}")

        # Якщо ми зустріли пост, старіший за нашу дату, або дійшли до ліміту API
        if last_post_date < stop_date:
            print(f" Для r/{subreddit} досягнуто часового ліміту.")
            break

        # Оновлюємо маркер 'after'
        after = data['data']['after']
        if not after:
            print(f" У r/{subreddit} сторінки закінчилися (ліміт API).")
            break

        # Пауза, щоб не заблокували
        time.sleep(1.5)

# 2. Збереження результатів у CSV
if all_posts:
    df = pd.DataFrame(all_posts)
    output_file = "reddit_data.csv"

    # Зберігаємо прямо в папку зі скриптом, з підтримкою укр літер
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print("-" * 30)
    print(f" Місія виконана!")
    print(f" Загалом зібрано постів: {len(df)}")
    print(f" Файл збережено: {output_file}")
else:
    print(" Не вдалося зібрати жодного поста.")