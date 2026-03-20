import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

# 1. Підготовка інструментів аналізу
nltk.download('punkt')
nltk.download('stopwords')
stop_words_en = set(stopwords.words('english'))


def tokenize_content(text):
    # Обробка порожніх значень, що критично для початкових етапів часового ряду
    if not isinstance(text, str) or text.strip() == "" or text.lower() == 'nan':
        return [], []

    # Очищення та приведення до нижнього регістру
    text_clean = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text_clean)

    # Видалення стоп-слів та шумів (токенів довжиною < 2)
    filtered = [w for w in tokens if w not in stop_words_en and len(w) > 1]

    # Генерація біграм для фіксації контекстних зв'язків (наприклад, "missile_strike")
    bi_grams = [" ".join(bg) for bg in ngrams(filtered, 2)]

    return filtered, bi_grams


# 2. Завантаження та обробка даних
df = pd.read_csv('features_final(1).csv')

# Створюємо єдине текстове поле, поєднуючи експертну аналітику та обговорення спільноти
df['total_text_context'] = df['isw_text'].fillna('') + " " + df['reddit_text'].fillna('')

print(" Запуск токенізації об'єднаного контексту...")

# Застосовуємо логіку обробки
results = df['total_text_context'].apply(tokenize_content)

df['unigrams'] = results.apply(lambda x: x[0])
df['bigrams'] = results.apply(lambda x: x[1])

# 3. Формування фінального набору ознак (Features)
# Залишаємо дату, погодні показники, цільову змінну та згенеровані токени
cols_to_keep = [
    'date',
    'target_alarm_next_day',
    'day_temp',
    'day_humidity',
    'day_conditions',
    'unigrams',
    'bigrams'
]

# Зберігаємо результат для подальшого навчання моделі
df[cols_to_keep].to_csv('tokenizer_data.csv', index=False, encoding='utf-8-sig')

print(f" Обробку завершено.")