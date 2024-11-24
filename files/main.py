import argparse
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from process import find_partners
from constants import MODEL_GROWTH_PATH, MODEL_GENETIC_VALU_PATH, MODEL_UPITANNOST_PATH, MODEL_INBREEDING_PATH, MODEL_HEALTH_PATH, MODEL_FERTILITY_PATH, MODEL_UDOY_PATH, ROOT_DIRECTORY, MERGED_DF_PATH, DF_ANIMALS_PATH

def main():
    # Создаем парсер аргументов
    parser = argparse.ArgumentParser(description="Find partner cows based on various criteria.")
    
    # Добавляем параметры для ввода
    parser.add_argument("target_cow_id", type=int, help="ID целевой коровы")
    parser.add_argument("category", type=str, help="Категория для поиска партнера (какой признак хотелось бы улучшить) (например, 'Прирост веса кг/день', Удой л/день', 'Здоровье (1-10)')")
    parser.add_argument("--min_growth", type=float, help="Минимальный прирост веса", default=None)
    parser.add_argument("--max_growth", type=float, help="Максимальный прирост веса", default=None)
    parser.add_argument("--min_udo", type=float, help="Минимальный удой", default=None)
    parser.add_argument("--max_udo", type=float, help="Максимальный удой", default=None)
    parser.add_argument("--min_health", type=int, help="Минимальное здоровье", default=None)
    parser.add_argument("--max_health", type=int, help="Максимальное здоровье", default=None)

    # Парсим аргументы
    args = parser.parse_args()
    # Загружаем модели
    model_growth = joblib.load(MODEL_GROWTH_PATH)
    model_genetic_value = joblib.load(MODEL_GENETIC_VALU_PATH)
    model_upitannost = joblib.load(MODEL_UPITANNOST_PATH)
    model_inbreeding = joblib.load(MODEL_INBREEDING_PATH)
    model_health = joblib.load(MODEL_HEALTH_PATH)
    model_fertility = joblib.load(MODEL_FERTILITY_PATH)
    model_udoy = joblib.load(MODEL_UDOY_PATH)
    
    # Загружаем DataFrame
    merged_df = pd.read_csv(ROOT_DIRECTORY / "datasets" / "merged_df.csv")
    df_animals = pd.read_csv(ROOT_DIRECTORY / "datasets" / "df_animals.csv")
    # Ищем партнеров
    result = find_partners(
        target_cow_id=args.target_cow_id,
        category=args.category,
        df_animals=df_animals,
        min_growth=args.min_growth,
        max_growth=args.max_growth,
        min_udo=args.min_udo,
        max_udo=args.max_udo,
        min_health=args.min_health,
        max_health=args.max_health
    )
    
    # Выводим результат
    print(result)

if __name__ == "__main__":
    main()
