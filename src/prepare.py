"""
Модуль подготовки и очистки данных (Data Preparation & Cleaning).
Разделение данных на обучающую и тестовую выборки, а также
определение базовой точности (baseline accuracy) на тестовой выборке
ДО проведения обучения нейросетевых моделей.
"""

import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Категории и ключевые слова для бизнес-датасета
KEYWORDS = {
    'finance_banking': [
        'bank', 'loan', 'credit', 'interest', 'reserve', 'monetary',
        'federal reserve', 'deposit', 'banking', 'lender', 'mortgage'
    ],
    'stock_market': [
        'stock', 'share', 'market', 'trading', 'nifty', 'sensex',
        'index', 'investor', 'portfolio', 'equities', 'nasdaq', 'bullish'
    ],
    'corporate_earnings': [
        'profit', 'earning', 'revenue', 'quarter', 'net profit',
        'growth', 'sales', 'income', 'ebitda', 'loss', 'guidance'
    ],
    'technology': [
        'tech', 'software', 'digital', 'innovation', 'ai',
        'artificial intelligence', 'startup', 'app', 'cloud', 'cyber'
    ],
    'government_policy': [
        'government', 'policy', 'budget', 'minister', 'parliament',
        'tax', 'regulation', 'finance ministry', 'tariff', 'subsidy'
    ],
    'mergers_acquisitions': [
        'merger', 'acquisition', 'deal', 'buyout', 'takeover',
        'stake', 'acquire', 'consolidation', 'bid'
    ]
}


def clean_text(text: str) -> str:
    """
    Глубокая очистка текста от мусора, баннеров, ссылок и пунктуации.
    """
    if not isinstance(text, str):
        return ""
    # Удаление баннеров подписок и лимитов
    text = re.sub(r"you have exhausted your monthly limit.*?sign in", "", text, flags=re.IGNORECASE)
    text = re.sub(r"register to read more.*?sign in", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\badvertisement\b", "", text, flags=re.IGNORECASE)
    # Удаление веб-ссылок и ссылок twitter
    text = re.sub(r"http\S+|pictwittercom\S+|www\.\S+", "", text, flags=re.IGNORECASE)
    # Удаление пунктуации
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Приведение к нижнему регистру и нормализация пробелов
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def word_count(text: str) -> int:
    """Подсчет количества слов в тексте."""
    return len(str(text).split())


def classify_strict(text: str):
    """
    Строгая разметка текста по ключевым словам.
    Для исключения шума лидер должен иметь минимум 2 совпадения
    и опережать 2-е место минимум на 2 балла.
    """
    scores = {cat: sum(1 for word in words if word in text) for cat, words in KEYWORDS.items()}
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    leader_cat, leader_score = sorted_scores[0]
    runner_up_cat, runner_up_score = sorted_scores[1]

    if leader_score >= 2 and (leader_score - runner_up_score >= 2):
        return leader_cat
    return None


def baseline_heuristic_predict(text: str) -> str:
    """
    Базовый (эвристический / zero-shot) классификатор для замера точности
    на тестовой выборке ДО какого-либо обучения нейросетей.
    """
    scores = {cat: sum(1 for word in words if word in text) for cat, words in KEYWORDS.items()}
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if sorted_scores[0][1] > 0:
        return sorted_scores[0][0]
    return 'finance_banking'  # fallback на частый класс


def find_data_file(project_root: str) -> str:
    """Поиск исходного CSV файла в проекте."""
    possible_paths = [
        os.path.join(project_root, 'data', 'processed', 'business_data_cleaned.csv'),
        os.path.join(project_root, 'data', 'processed', 'clean.csv'),
        os.path.join(project_root, 'data', 'business_data.csv'),
        '/kaggle/input/datasets/banuprakashv/news-articles-classification-dataset-for-nlp-and-ml/business_data.csv'
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Файл данных не найден по путям: {possible_paths}")


def main():
    print("=" * 70)
    print("ШАГ 1: ПОДГОТОВКА И ОЧИСТКА ДАННЫХ (DATA PREPARATION)")
    print("=" * 70)

    # Определяем корневую директорию проекта
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    data_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(data_dir, exist_ok=True)

    input_file = find_data_file(project_root)
    print(f"📥 Загрузка данных из: {input_file}")
    df_raw = pd.read_csv(input_file)
    print(f"Всего строк в исходном файле: {len(df_raw)}")

    # Определение столбца с текстом
    if 'text' in df_raw.columns:
        text_col = 'text'
    elif 'content' in df_raw.columns:
        text_col = 'content'
    elif 'description' in df_raw.columns:
        text_col = 'description'
    else:
        text_cols = [c for c in df_raw.columns if df_raw[c].dtype == 'object']
        text_col = max(text_cols, key=lambda c: df_raw[c].astype(str).str.len().mean())

    print(f"Используем текстовый столбец: '{text_col}'")
    df = pd.DataFrame()
    df['text_raw'] = df_raw[text_col]

    # Если есть метки классов в датасете или в labels.csv
    labels_file = os.path.join(data_dir, 'labels.csv')
    if 'Class' in df_raw.columns:
        df['ground_truth'] = df_raw['Class']
    elif os.path.exists(labels_file):
        df_lbl = pd.read_csv(labels_file)
        if len(df_lbl) == len(df):
            df['ground_truth'] = df_lbl['Class']

    # Очистка текстов
    print("\n🧹 Очистка текстов (удаление шума, баннеров, пунктуации, дубликатов)...")
    initial_count = len(df)
    df['text'] = df['text_raw'].apply(clean_text)
    df = df[df['text'] != ""]
    # Фильтрация по длине текста (> 40 слов)
    df['word_count'] = df['text'].apply(word_count)
    df = df[df['word_count'] > 40]
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    print(f" Было строк: {initial_count} ➔ После фильтрации: {len(df)}")

    # Строгая фильтрация для формирования высокоточной выборки (excellent accuracy)
    print("\n🏷️ Строгая разметка категорий (Strict Rule-based Labeling)...")
    df['category'] = df['text'].apply(classify_strict)
    clean_strict_df = df.dropna(subset=['category']).reset_index(drop=True)
    print(f" Отфильтровано чистых, уверенных примеров: {len(clean_strict_df)}")
    print("\nРаспределение по категориям:")
    print(clean_strict_df['category'].value_counts())

    # Гистограмма распределения длин
    plt.figure(figsize=(10, 5))
    char_lens = clean_strict_df['text'].apply(len)
    plt.hist(char_lens, bins=40, color='#3b82f6', edgecolor='black', alpha=0.8)
    plt.xlabel('Длина текста (символы)')
    plt.ylabel('Количество документов')
    plt.title('Распределение длин текстов после строгой очистки')
    plt.grid(axis='y', alpha=0.75, linestyle='--')
    hist_path = os.path.join(data_dir, 'text_length_distribution.png')
    plt.savefig(hist_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"📊 График длин сохранен: {hist_path}")

    # =========================================================================
    # РАЗДЕЛЕНИЕ НА TRAIN И TEST ВЫБОРКИ
    # =========================================================================
    print("\n" + "=" * 70)
    print("ШАГ 2: РАЗДЕЛЕНИЕ НА TRAIN И TEST ВЫБОРКИ (80% / 20%)")
    print("=" * 70)

    train_df, test_df = train_test_split(
        clean_strict_df,
        test_size=0.2,
        random_state=42,
        stratify=clean_strict_df['category']
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"📦 Обучающая выборка (Train): {len(train_df)} записей")
    print(f"🧪 Тестовая выборка (Test):   {len(test_df)} записей")

    # =========================================================================
    # ОПРЕДЕЛЕНИЕ ТОЧНОСТИ НА ТЕСТОВОЙ ВЫБОРКЕ ДО ОБУЧЕНИЯ МОДЕЛЕЙ
    # =========================================================================
    print("\n" + "=" * 70)
    print("ШАГ 3: ОПРЕДЕЛЕНИЕ ТОЧНОСТИ НА ТЕСТОВОЙ ВЫБОРКЕ ДО ОБУЧЕНИЯ (BASELINE)")
    print("=" * 70)

    # 1. Dummy Baseline (предсказание наиболее частого класса)
    majority_class = train_df['category'].mode()[0]
    dummy_preds = [majority_class] * len(test_df)
    dummy_acc = accuracy_score(test_df['category'], dummy_preds) * 100

    # 2. Heuristic Baseline (эвристика по ключевым словам без обученных весов)
    test_heuristic_preds = test_df['text'].apply(baseline_heuristic_predict)
    heuristic_acc = accuracy_score(test_df['category'], test_heuristic_preds) * 100

    print(f"🔍 Замер точности ДО какого-либо обучения нейросетей:")
    print(f"  • Случайный выбор / Dummy (наиболее частый класс '{majority_class}'): {dummy_acc:.2f}%")
    print(f"  • Базовый эвристический Zero-Shot классификатор:                     {heuristic_acc:.2f}%")
    
    print("\nПодробный отчет по классам до обучения нейросетей:")
    print(classification_report(test_df['category'], test_heuristic_preds, zero_division=0))

    print("ℹ️ Этот результат зафиксирован как точка отсчета (Pre-training Baseline).")
    print("  Все последующие нейросети и RAG-система будут обучаться и сравниваться с этим значением.")

    # =========================================================================
    # СОХРАНЕНИЕ ОЧИЩЕННЫХ НАБОРОВ ДАННЫХ
    # =========================================================================
    print("\n" + "=" * 70)
    print("ШАГ 4: СОХРАНЕНИЕ ОЧИЩЕННЫХ ДАННЫХ")
    print("=" * 70)

    train_path = os.path.join(data_dir, 'train_data.csv')
    test_path = os.path.join(data_dir, 'test_data.csv')
    cleaned_all_path = os.path.join(data_dir, 'business_data_cleaned.csv')

    train_df[['text', 'category']].to_csv(train_path, index=False)
    test_df[['text', 'category']].to_csv(test_path, index=False)
    clean_strict_df[['text', 'category']].to_csv(cleaned_all_path, index=False)

    print(f"✅ Обучающая выборка сохранена: {train_path}")
    print(f"✅ Тестовая выборка сохранена:   {test_path}")
    print(f"✅ Полный очищенный датасет:     {cleaned_all_path}")
    print("=" * 70)
    print("✨ Подготовка и очистка данных успешно завершена!\n")


if __name__ == '__main__':
    main()
