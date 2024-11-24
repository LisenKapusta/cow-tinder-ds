import pandas as pd
import numpy as np
import tqdm
from pathlib import Path
import joblib
from datetime import datetime

from constants import MODEL_GROWTH_PATH, MODEL_GENETIC_VALU_PATH, MODEL_UPITANNOST_PATH, MODEL_INBREEDING_PATH, MODEL_HEALTH_PATH, MODEL_FERTILITY_PATH, MODEL_UDOY_PATH, ROOT_DIRECTORY, MERGED_DF_PATH, DF_ANIMALS_PATH
pd.set_option('display.max_columns', None)
merged_df = pd.read_csv(ROOT_DIRECTORY / "datasets" / "merged_df.csv")
df_animals = pd.read_csv(ROOT_DIRECTORY / "datasets" / "df_animals.csv")


model_growth = joblib.load(MODEL_GROWTH_PATH)
model_genetic_value = joblib.load(MODEL_GENETIC_VALU_PATH)
model_upitannost = joblib.load(MODEL_UPITANNOST_PATH)
model_inbreeding = joblib.load(MODEL_INBREEDING_PATH)
model_health = joblib.load(MODEL_HEALTH_PATH)
model_fertility = joblib.load(MODEL_FERTILITY_PATH)
model_udoy = joblib.load(MODEL_UDOY_PATH)

def find_partners(target_cow_id, category, df_animals=df_animals, min_growth=None, max_growth=None, min_udo=None, max_udo=None, min_health=None, max_health=None):
    # Фильтрация: исключение самой коровы
    data = df_animals[df_animals['ID_особи'] != target_cow_id]
    target_cow = df_animals[df_animals['ID_особи'] == target_cow_id].iloc[0]
    target_cow_gender = target_cow['Пол']

    target_cow_father = target_cow['Родитель_папа']
    target_cow_mother = target_cow['Родитель_мама']
    # Находим прямых родственников (общие родители)
    relatives = set(data[(data['Родитель_папа'] == target_cow_father) | 
                        (data['Родитель_мама'] == target_cow_mother)]['ID_особи'])

    # Исключаем родственников из данных
    data = data[~data['ID_особи'].isin(relatives)]
    # Определяем пол целевой коровы
    target_cow_gender = target_cow['Пол']

    # Устанавливаем противоположный пол
    if target_cow_gender == 'Самка':
        opposite_gender = 'Самец'
    else:
        opposite_gender = 'Самка'

    # Рассчитываем возраст (в месяцах)
    current_date = datetime.now()
    data['Возраст_месяцев'] = (current_date - pd.to_datetime(data['Дата_Рождения'])).dt.days // 30

    # Фильтрация по противоположному полу и возрасту
    data = data[(data['Пол'] == opposite_gender) & (data['Возраст_месяцев'] >= 18) & (data['Возраст_месяцев'] <= 24)]
    # Фильтрация по желаемым диапазонам
    
    # Пример: расчет ожидаемого коэффициента инбридинга
    data["offspring_inbreeding_coefficient"] = (target_cow["Коэффициент инбридинга (F)"] + 
                                        data["Коэффициент инбридинга (F)"]) / 2
    data = data.sort_values(by="offspring_inbreeding_coefficient", ascending=True)
    threshold_inbreeding = 0.1
    data = data[data['offspring_inbreeding_coefficient'] < threshold_inbreeding]

    def cow(category, cow_id, merged_df=merged_df):
        id_mask = merged_df["ID_особи"] == cow_id
        category_mask = merged_df["Признак"] == category

        category_data = merged_df[id_mask & category_mask]

        if not category_data.empty:
            # Суммируем эффекты мутации для всех мутаций для данной категории
            total_mutation_effect = category_data["mutation_effect"].sum()
            return total_mutation_effect
        else:
            return 0

    data["mutation_effect"] = data["ID_особи"].apply(lambda x: cow(category, x))

   
    def predict_genetic_value(родитель_папа, родитель_мама, model_genetic_value, df):
        # Данные о родителях
        parent_data = df[(df["ID_особи"] == родитель_папа) | (df["ID_особи"] == родитель_мама)]
        

        if parent_data["ID_особи"].nunique() < 2:
            return "Не найдены оба родителя в таблице!"

        мама = parent_data[parent_data["ID_особи"] == родитель_мама].iloc[0]
        папа = parent_data[parent_data["ID_особи"] == родитель_папа].iloc[0]
        
        # Формируем признаки для предсказания
        input_data = pd.DataFrame([{
            "Прирост веса кг/день_мама": мама["Прирост веса кг/день"],
            "Фертильность (%)_мама": мама["Фертильность (%)"],
            "Здоровье (1-10)_мама": мама["Здоровье (1-10)"],
            "Упитанность_мама": мама["Упитанность"],
            "Прирост веса кг/день_папа": папа["Прирост веса кг/день"],
            "Фертильность (%)_папа": папа["Фертильность (%)"],
            "Здоровье (1-10)_папа": папа["Здоровье (1-10)"],
            "Упитанность_папа": папа["Упитанность"]
        }])
        # Предсказание генетической ценности
        predicted_genetic_value = model_genetic_value.predict(input_data)[0]
        
        # Формирование данных о потомке
        offspring = predicted_genetic_value
        return offspring


    def predict_growth(родитель_папа, родитель_мама, model_growth, df):
        # Данные о родителях
        parent_data = df[(df["ID_особи"] == родитель_папа) | (df["ID_особи"] == родитель_мама)]
        
        if parent_data["ID_особи"].nunique() < 2:
            return "Не найдены оба родителя в таблице!"
        
        мама = parent_data[parent_data["ID_особи"] == родитель_мама].iloc[0]
        папа = parent_data[parent_data["ID_особи"] == родитель_папа].iloc[0]
        
        # Формируем признаки для предсказания
        input_data = pd.DataFrame([{
            "Упитанность_мама": мама["Упитанность"],
            "Коэффициент инбридинга (F)_мама": мама["Коэффициент инбридинга (F)"],
            "Прирост веса кг/день_мама": мама["Прирост веса кг/день"],
            "Здоровье (1-10)_мама": мама["Здоровье (1-10)"],
            "Фертильность (%)_мама": мама["Фертильность (%)"],
            "Генетическая ценность (баллы)_мама": мама["Генетическая ценность (баллы)"],
            "Упитанность_папа": папа["Упитанность"],
            "Коэффициент инбридинга (F)_папа": папа["Коэффициент инбридинга (F)"],
            "Прирост веса кг/день_папа": папа["Прирост веса кг/день"],
            "Здоровье (1-10)_папа": папа["Здоровье (1-10)"],
            "Фертильность (%)_папа": папа["Фертильность (%)"],
            "Генетическая ценность (баллы)_папа": папа["Генетическая ценность (баллы)"]
        }])
        
        # Предсказание прироста массы
        predicted_growth = model_growth.predict(input_data)[0]
        
        # Формирование данных о потомке
        offspring = predicted_growth
        return offspring

    def predict_fertility(родитель_папа, родитель_мама, model_fertility, df):
        # Данные о родителях
        parent_data = df[(df["ID_особи"] == родитель_папа) | (df["ID_особи"] == родитель_мама)]
        
        if parent_data["ID_особи"].nunique() < 2:
            return "Не найдены оба родителя в таблице!"
        
        мама = parent_data[parent_data["ID_особи"] == родитель_мама].iloc[0]
        папа = parent_data[parent_data["ID_особи"] == родитель_папа].iloc[0]
        
        # Формируем признаки для предсказания
        input_data = pd.DataFrame([{
            "Упитанность_мама": мама["Упитанность"],
            "Коэффициент инбридинга (F)_мама": мама["Коэффициент инбридинга (F)"],
            "Прирост веса кг/день_мама": мама["Прирост веса кг/день"],
            "Здоровье (1-10)_мама": мама["Здоровье (1-10)"],
            "Фертильность (%)_мама": мама["Фертильность (%)"],
            "Генетическая ценность (баллы)_мама": мама["Генетическая ценность (баллы)"],
            "Упитанность_папа": папа["Упитанность"],
            "Коэффициент инбридинга (F)_папа": папа["Коэффициент инбридинга (F)"],
            "Прирост веса кг/день_папа": папа["Прирост веса кг/день"],
            "Здоровье (1-10)_папа": папа["Здоровье (1-10)"],
            "Фертильность (%)_папа": папа["Фертильность (%)"],
            "Генетическая ценность (баллы)_папа": папа["Генетическая ценность (баллы)"]
        }])
        
        # Предсказание фертильности
        predicted_fertility = model_fertility.predict(input_data)[0]
        
        # Формирование данных о потомке
        offspring = predicted_fertility
        return offspring

    def predict_health(родитель_папа, родитель_мама, model_health, df):
        # Данные о родителях
        parent_data = df[(df["ID_особи"] == родитель_папа) | (df["ID_особи"] == родитель_мама)]
        
        if parent_data["ID_особи"].nunique() < 2:
            return "Не найдены оба родителя в таблице!"
        
        мама = parent_data[parent_data["ID_особи"] == родитель_мама].iloc[0]
        папа = parent_data[parent_data["ID_особи"] == родитель_папа].iloc[0]
        
        # Формируем признаки для предсказания
        input_data = pd.DataFrame([{
            "Упитанность_мама": мама["Упитанность"],
            "Коэффициент инбридинга (F)_мама": мама["Коэффициент инбридинга (F)"],
            "Прирост веса кг/день_мама": мама["Прирост веса кг/день"],
            "Фертильность (%)_мама": мама["Фертильность (%)"],
            "Здоровье (1-10)_мама": мама["Здоровье (1-10)"],
            "Генетическая ценность (баллы)_мама": мама["Генетическая ценность (баллы)"],
            "Упитанность_папа": папа["Упитанность"],
            "Коэффициент инбридинга (F)_папа": папа["Коэффициент инбридинга (F)"],
            "Прирост веса кг/день_папа": папа["Прирост веса кг/день"],
            "Фертильность (%)_папа": папа["Фертильность (%)"],
            "Здоровье (1-10)_папа": папа["Здоровье (1-10)"],
            "Генетическая ценность (баллы)_папа": папа["Генетическая ценность (баллы)"]
        }])
        
        # Предсказание здоровья
        predicted_health = model_health.predict(input_data)[0]
        
        # Формирование данных о потомке
        offspring = predicted_health
        return offspring

    def predict_inbreeding(родитель_папа, родитель_мама, model_inbreeding, df):
        # Данные о родителях
        parent_data = df[(df["ID_особи"] == родитель_папа) | (df["ID_особи"] == родитель_мама)]
        
        if parent_data["ID_особи"].nunique() < 2:
            return "Не найдены оба родителя в таблице!"
        
        мама = parent_data[parent_data["ID_особи"] == родитель_мама].iloc[0]
        папа = parent_data[parent_data["ID_особи"] == родитель_папа].iloc[0]
        
        # Формируем признаки для предсказания
        input_data = pd.DataFrame([{
            "Упитанность_мама": мама["Упитанность"],
            "Прирост веса кг/день_мама": мама["Прирост веса кг/день"],
            "Фертильность (%)_мама": мама["Фертильность (%)"],
            "Здоровье (1-10)_мама": мама["Здоровье (1-10)"],
            "Генетическая ценность (баллы)_мама": мама["Генетическая ценность (баллы)"],
            "Упитанность_папа": папа["Упитанность"],
            "Прирост веса кг/день_папа": папа["Прирост веса кг/день"],
            "Фертильность (%)_папа": папа["Фертильность (%)"],
            "Здоровье (1-10)_папа": папа["Здоровье (1-10)"],
            "Генетическая ценность (баллы)_папа": папа["Генетическая ценность (баллы)"]
        }])
        
        # Предсказание коэффициента инбридинга (F)
        predicted_inbreeding = model_inbreeding.predict(input_data)[0]
        
        # Формирование данных о потомке
        offspring = predicted_inbreeding
        return offspring

    def predict_upitannost(родитель_папа, родитель_мама, model_upitannost, df):
        # Данные о родителях
        parent_data = df[(df["ID_особи"] == родитель_папа) | (df["ID_особи"] == родитель_мама)]
        
        if parent_data["ID_особи"].nunique() < 2:
            return "Не найдены оба родителя в таблице!"
        
        мама = parent_data[parent_data["ID_особи"] == родитель_мама].iloc[0]
        папа = parent_data[parent_data["ID_особи"] == родитель_папа].iloc[0]
        
        # Формируем признаки для предсказания
        input_data = pd.DataFrame([{
            "Прирост веса кг/день_мама": мама["Прирост веса кг/день"],
            "Фертильность (%)_мама": мама["Фертильность (%)"],
            "Здоровье (1-10)_мама": мама["Здоровье (1-10)"],
            "Генетическая ценность (баллы)_мама": мама["Генетическая ценность (баллы)"],
            "Прирост веса кг/день_папа": папа["Прирост веса кг/день"],
            "Фертильность (%)_папа": папа["Фертильность (%)"],
            "Здоровье (1-10)_папа": папа["Здоровье (1-10)"],
            "Генетическая ценность (баллы)_папа": папа["Генетическая ценность (баллы)"]
        }])
        
        # Предсказание упитанности
        predicted_upitannost = model_upitannost.predict(input_data)[0]
        
        # Формирование данных о потомке
        offspring = predicted_upitannost
        return offspring

    def predict_udoy(родитель_папа, родитель_мама, model_udoy, df):
        # Данные о родителях
        parent_data = df[(df["ID_особи"] == родитель_папа) | (df["ID_особи"] == родитель_мама)]
        
        if parent_data["ID_особи"].nunique() < 2:
            return "Не найдены оба родителя в таблице!"
        
        мама = parent_data[parent_data["ID_особи"] == родитель_мама].iloc[0]
        папа = parent_data[parent_data["ID_особи"] == родитель_папа].iloc[0]
        
        # Формируем признаки для предсказания
        input_data = pd.DataFrame([{
            "Упитанность_мама": мама["Упитанность"],
            "Коэффициент инбридинга (F)_мама": мама["Коэффициент инбридинга (F)"],
            "Прирост веса кг/день_мама": мама["Прирост веса кг/день"],
            "Здоровье (1-10)_мама": мама["Здоровье (1-10)"],
            "Фертильность (%)_мама": мама["Фертильность (%)"],
            "Генетическая ценность (баллы)_мама": мама["Генетическая ценность (баллы)"],
            "Упитанность_папа": папа["Упитанность"],
            "Коэффициент инбридинга (F)_папа": папа["Коэффициент инбридинга (F)"],
            "Прирост веса кг/день_папа": папа["Прирост веса кг/день"],
            "Здоровье (1-10)_папа": папа["Здоровье (1-10)"],
            "Фертильность (%)_папа": папа["Фертильность (%)"],
            "Генетическая ценность (баллы)_папа": папа["Генетическая ценность (баллы)"]
        }])
        
        # Предсказание удоя
        predicted_udoy = model_udoy.predict(input_data)[0]
        
        # Формирование данных о потомке
        offspring = predicted_udoy
        return offspring
    
    def parents(another_cow_id):
        if target_cow["Пол"] == "Самка":
            родитель_мама = target_cow["ID_особи"]
            родитель_папа = another_cow_id
            return родитель_мама, родитель_папа

        родитель_мама = another_cow_id
        родитель_папа = target_cow["ID_особи"]
        return родитель_мама, родитель_папа

    df_result = data
    df_result["parents"] = df_result["ID_особи"].apply(lambda x: parents(x))

    df_result["child_ID_особи"] = df_result["parents"].apply(lambda x: f"{x[0]}_{x[1]}_child")



    df_result["Прирост веса кг/день потомка"] = df_result["parents"].apply(lambda x: predict_growth(родитель_папа=x[1], родитель_мама=x[0], model_growth=model_growth, df=merged_df))
    df_result["Генетическая ценность (баллы) потомка"] = df_result["parents"].apply(lambda x: predict_genetic_value(родитель_папа=x[1], родитель_мама=x[0], model_genetic_value=model_genetic_value, df=merged_df))
    df_result["Упитанность потомка"] = df_result["parents"].apply(lambda x: predict_upitannost(родитель_папа=x[1], родитель_мама=x[0], model_upitannost=model_upitannost, df=merged_df))
    df_result["Коэффициент инбридинга (F) потомка"] = df_result["parents"].apply(lambda x: predict_inbreeding(родитель_папа=x[1], родитель_мама=x[0], model_inbreeding=model_inbreeding, df=merged_df))
    df_result["Здоровье (1-10) потомка"] = df_result["parents"].apply(lambda x: predict_health(родитель_папа=x[1], родитель_мама=x[0], model_health=model_health, df=merged_df))
    df_result["Фертильность (%) потомка"] = df_result["parents"].apply(lambda x: predict_fertility(родитель_папа=x[1], родитель_мама=x[0], model_fertility=model_fertility, df=merged_df))
    df_result["Удой л/день потомка"] = df_result["parents"].apply(lambda x: predict_udoy(родитель_папа=x[1], родитель_мама=x[0], model_udoy=model_udoy, df=merged_df))

    df_result = df_result[
    ~df_result['Прирост веса кг/день потомка'].isin(['Не найдены оба родителя в таблице!']) &
    ~df_result['Удой л/день потомка'].isin(['Не найдены оба родителя в таблице!']) &
    ~df_result['Здоровье (1-10) потомка'].isin(['Не найдены оба родителя в таблице!'])
]

    
    if min_growth is not None:
        df_result = df_result[df_result['Прирост веса кг/день потомка'] >= min_growth]
    if max_growth is not None:
        df_result = df_result[df_result['Прирост веса кг/день потомка'] <= max_growth]
    if min_udo is not None:
        df_result = df_result[df_result['Удой л/день потомка'] >= min_udo]
    if max_udo is not None:
        df_result = df_result[df_result['Удой л/день потомка'] <= max_udo]
    if min_health is not None:
        df_result = df_result[df_result['Здоровье (1-10) потомка'] >= min_health]
    if max_health is not None:
        df_result = df_result[df_result['Здоровье (1-10) потомка'] <= max_health]


    sorted_data = df_result.sort_values(by=["mutation_effect", "offspring_inbreeding_coefficient"], ascending=[False, True])

    sorted_data = sorted_data.loc[:, ['ID_особи', 'Пол', 'Порода', 'Дата_Рождения', 'Родитель_папа',
        'Родитель_мама', 'Удой л/день', 'Упитанность',
        'Коэффициент инбридинга (F)', 'Прирост веса кг/день', 'Здоровье (1-10)',
        'Фертильность (%)', 'Генетическая ценность (баллы)',
        'child_ID_особи', 'Прирост веса кг/день потомка',
        'Генетическая ценность (баллы) потомка', 'Упитанность потомка',
        'Коэффициент инбридинга (F) потомка', 'Здоровье (1-10) потомка',
        'Фертильность (%) потомка', "Удой л/день потомка"]]

    top_5_results = sorted_data.head(5)
    

    return top_5_results
   