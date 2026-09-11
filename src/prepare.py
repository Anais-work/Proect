import re
import pandas as pd
import matplotlib.pyplot as plt  

file_path = '/kaggle/input/datasets/banuprakashv/news-articles-classification-dataset-for-nlp-and-ml/business_data.csv'
print(f"Загрузка датасета...")
df_raw = pd.read_csv(file_path)
print(f"Загружено строк: {len(df_raw)}")

def clean_and_lowercase(text):
    if not isinstance(text, str):
        return ""
    text_lower = text.lower()
    cleaned = re.sub(r'[^a-z0-9\s]', '', text_lower)
    single_spaced = re.sub(r'\s+', ' ', cleaned)
    return single_spaced.strip()

def word_count(text):
    return len(str(text).split())

if 'content' in df_raw.columns:
    main_col = 'content'
elif 'description' in df_raw.columns:
    main_col = 'description'
else:
    text_cols = [col for col in df_raw.columns if df_raw[col].dtype == 'object']
    main_col = max(text_cols, key=lambda col: df_raw[col].astype(str).str.len().mean())

print(f"Работаем со столбцом: '{main_col}'")
df = df_raw[[main_col]].rename(columns={main_col: 'text'})

print("\nОчистка данных...")
initial_len = len(df)
df = df.dropna(subset=['text'])
df['text'] = df['text'].apply(clean_and_lowercase)
df = df[df['text'] != ""]
df = df.drop_duplicates(subset=['text'])
df = df[df['text'].apply(word_count) > 50]
df = df.reset_index(drop=True)

print(f" Было: {initial_len} → Стало: {len(df)} строк")

print("\n Шаг 3: Анализ длин текстов (символы)...")
df['char_length'] = df['text'].apply(len)

# Статистика
print(df['char_length'].describe())

# Гистограмма
plt.figure(figsize=(10, 5))
plt.hist(df['char_length'], bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Длина текста (символы)')
plt.ylabel('Количество записей')
plt.title('Распределение длин текстов после очистки')
plt.grid(axis='y', alpha=0.75)
plt.savefig('/kaggle/working/text_length_distribution.png') 
plt.show()

print("\nКлассификация текстов...")
keywords = {
    'finance_banking': ['bank', 'loan', 'credit', 'interest', 'reserve', 'monetary', 'federal reserve', 'deposit'],
    'stock_market': ['stock', 'share', 'market', 'trading', 'nifty', 'sensex', 'index', 'investor', 'portfolio'],
    'corporate_earnings': ['profit', 'earning', 'revenue', 'quarter', 'net profit', 'growth', 'sales', 'income'],
    'technology': ['tech', 'software', 'digital', 'innovation', 'ai', 'artificial intelligence', 'startup', 'app'],
    'government_policy': ['government', 'policy', 'budget', 'minister', 'parliament', 'tax', 'regulation', 'finance ministry'],
    'mergers_acquisitions': ['merger', 'acquisition', 'deal', 'buyout', 'takeover', 'stake', 'acquire']
}

def classify_by_keywords(text):
    text = text.lower()
    scores = {}
    for category, words in keywords.items():
        score = sum(1 for word in words if word in text)
        scores[category] = score
    
    max_score = max(scores.values())
    if max_score == 0:
        return 'other'
    
    for category, score in scores.items():
        if score == max_score:
            return category

df['Class'] = df['text'].apply(classify_by_keywords)
print(f"\n Классификация завершена!")
print("\n Распределение по классам:")
print(df['Class'].value_counts())

output_file = '/kaggle/working/clean.csv'
df.to_csv(output_file, index=False)
print(f"\n Файл сохранен: {output_file}")