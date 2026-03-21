import pandas as pd
import numpy as np
import os

alerts = pd.read_csv('1_alerts.csv')
isw = pd.read_csv('2_isw.csv')
weather = pd.read_csv('3_weather.csv')

reddit_path = 'дані/reddit_data.csv' if os.path.exists('дані/reddit_data.csv') else 'reddit_data.csv'
reddit = pd.read_csv(reddit_path)

region_map = {
    'Львівська обл.': 'Lviv,Ukraine', 'Чернігівська обл.': 'Chernihiv,Ukraine',
    'Вінницька обл.': 'Vinnytsia,Ukraine', 'Харківська обл.': 'Kharkiv,Ukraine',
    'Тернопільська обл.': 'Ternopil,Ukraine', 'Київ': 'Kyiv,Ukraine',
    'Рівненська обл.': 'Rivne,Ukraine', 'Черкаська обл.': 'Cherkasy,Ukraine',
    'Одеська обл.': 'Odesa,Ukraine', 'Запорізька обл.': 'Zaporozhye,Ukraine',
    'Волинська обл.': 'Lutsk,Ukraine', 'Житомирська обл.': 'Zhytomyr,Ukraine',
    'Херсонська обл.': 'Kherson,Ukraine', 'Миколаївська обл.': 'Mykolaiv,Ukraine',
    'Хмельницька обл.': 'Khmelnytskyi,Ukraine', 'Івано-Франківська обл.': 'Ivano-Frankivsk,Ukraine',
    'Дніпропетровська обл.': 'Dnipro,Ukraine', 'Кіровоградська обл.': 'Kropyvnytskyi,Ukraine',
    'Чернівецька обл.': 'Chernivtsi,Ukraine', 'Полтавська обл.': 'Poltava,Ukraine',
    'Київська обл.': 'Kyiv,Ukraine', 'Сумська обл.': 'Sumy,Ukraine',
    'Донецька обл.': 'Donetsk,Ukraine', 'Закарпатська обл.': 'Uzhgorod,Ukraine',
}

alerts['date'] = pd.to_datetime(alerts['start']).dt.date.astype(str)
alerts['duration_min'] = (pd.to_datetime(alerts['end']) - pd.to_datetime(alerts['start'])).dt.total_seconds() / 60
alerts_agg = alerts.groupby(['date', 'region_city']).agg(
    alarm_count=('region_city', 'count'),
    alarm_total_duration_min=('duration_min', 'sum'),
    alarm_all_region=('all_region', 'max')
).reset_index()

all_dates = alerts_agg['date'].unique()
all_regions = list(region_map.keys())
df = pd.DataFrame([(d, r) for d in all_dates for r in all_regions], columns=['date', 'region_city'])
df = df.merge(alerts_agg, on=['date', 'region_city'], how='left').fillna(0)
df['city_address'] = df['region_city'].map(region_map)

# weather
weather_day = weather.drop_duplicates(subset=['city_address', 'day_datetime']).rename(columns={'day_datetime': 'date'})
weather_day['date'] = weather_day['date'].astype(str)
cols_w = ['city_address', 'date', 'day_tempmax', 'day_tempmin', 'day_temp', 'day_humidity', 'day_windspeed', 'day_conditions']
df = df.merge(weather_day[cols_w], on=['date', 'city_address'], how='left')

# isw
isw['date'] = isw['date'].astype(str)
df = df.merge(isw[['date', 'text']].rename(columns={'text': 'isw_text'}), on='date', how='left').fillna({'isw_text': ''})

# reddit
reddit['date'] = pd.to_datetime(reddit['date']).dt.date.astype(str)
reddit_agg = reddit.groupby('date').agg(
    reddit_post_count=('title', 'count'),
    reddit_avg_score=('score', 'mean'),
    reddit_text=('title', lambda x: ' '.join(x.dropna()))
).reset_index()
df = df.merge(reddit_agg, on='date', how='left').fillna({'reddit_post_count': 0, 'reddit_avg_score': 0, 'reddit_text': ''})

# Таргет та фічі
df = df.sort_values(['region_city', 'date'])
df['target_alarm_next_day'] = (df.groupby('region_city')['alarm_count'].shift(-1) > 0).astype(float)
df = df.dropna(subset=['target_alarm_next_day'])

df.to_csv('features_final.csv', index=False)
print('Успіх! Файл features_final.csv створено:', df.shape)