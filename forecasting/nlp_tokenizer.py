import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams



def prepare_nltk():
    resources = ['punkt', 'punkt_tab', 'stopwords']
    for res in resources:
        try:
            if res == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif res == 'punkt':
                nltk.data.find('tokenizers/punkt')
            else:
                nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print(f"Завантаження ресурсу: {res}...")
            nltk.download(res)


prepare_nltk()


def auto_tokenize_dataframe(file_path):
    print(f"Читання файлу: {file_path}")
    df = pd.read_csv(file_path)

    # Знаходимо текстові стовпці, АЛЕ ігноруємо ті, що містять 'date' або 'id'
    # Це допоможе уникнути помилок на стовпці 'date'
    text_cols = [col for col in df.select_dtypes(include=['object']).columns
                 if 'date' not in col.lower() and 'id' not in col.lower()]

    print(f"Буде оброблено текстові стовпці: {text_cols}")

    stop_words = set(stopwords.words('english'))

    def process_text(text):
        if not isinstance(text, str) or text.strip() == "":
            return None, None

        # Очищення від символів (тільки англійські літери)
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = word_tokenize(text)

        # Видалення стоп-слів (Task 5.3)
        filtered = [w for w in tokens if w not in stop_words and len(w) > 1]

        # Створення біграм (Task 5.3)
        bg = [" ".join(pair) for pair in ngrams(filtered, 2)]

        return filtered, bg

    for col in text_cols:
        print(f"Токенізація стовпця: {col}...")
        results = df[col].fillna('').apply(process_text)

        df[f'{col}_unigrams'] = results.apply(lambda x: x[0] if x else [])
        df[f'{col}_bigrams'] = results.apply(lambda x: x[1] if x else [])

    return df


if __name__ == "__main__":
    input_file = "../data_receiver/merged_data.csv"
    final_df = auto_tokenize_dataframe(input_file)

    output_name = "tokenized_data.csv"
    final_df.to_csv(output_name, index=False)
    print(f"\nГотово! Результат у файлі: {output_name}")