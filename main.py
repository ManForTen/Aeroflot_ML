import streamlit as st
import joblib
import datetime
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
st.write("""
# Прогноз количества мест на рейсе
""")
mod1, mod2 = st.tabs(["1 Модель (Bagging Regressor)", "2 Модель (Hist Gradient Boosting Regressor)"])
# Список названий моделей
model_names = [
    'Y',
    'P',
    'E',
    'B',
    'K',
    'L',
    'V',
    'Q',
    'X',
    'M',
    'J',
    'C',
    'D',
    'O',
    'H',
    'Z',
    'T',
    'I',
    'N',
    'G',
    'R',
    'U'
]

predictions_dict = {}  # Словарь для хранения предсказаний для каждой модели


def transform_date(date):
    origin = datetime.datetime(1970, 1, 1)

    if isinstance(date, datetime.date):  # Добавлено условие для datetime.date
        date = datetime.datetime.combine(date, datetime.datetime.min.time())

    # Преобразование даты в дни с начала отсчета (origin)
    days_from_origin = (date - origin).days

    # Определение выходного дня
    weekend = 1 if date.weekday() > 4 else 0

    # День недели
    day_of_week = date.weekday()

    # Праздник
    apr_days, may_days = [29, 30], [1, 6, 7, 8, 9]
    dec_days, jan_days = [29, 30, 31], [1, 2, 3, 4, 5, 6, 7, 8]
    april, may, december, january = 4, 5, 12, 1

    holidays = (
            (date.day in apr_days and date.month == april) or
            (date.day in may_days and date.month == may) or
            (date.day in dec_days and date.month == december) or
            (date.day in jan_days and date.month == january)
    )

    return {
        'FLTDAT': days_from_origin,
        'weekend': weekend,
        'dayOfWeek': day_of_week + 1,
        'holidays': int(holidays)
    }


input_date = st.date_input("Выберите дату:", datetime.datetime(2019, 12, 31))
transformed_data = transform_date(input_date)

flt_time = st.time_input('Выберите время', value=datetime.time(13, 35))

new_data = pd.DataFrame({
    'SA': [0.0],  # По умолчанию
    'AU': [4.0],  # По умолчанию
    'DTD': [st.number_input('Введите количество дней до вылета рейса:', min_value=0)],  # Ввод пользователя
    'FLTNUM': [st.number_input('Введите № рейса:', min_value=0)],  # Ввод пользователя
    'fltHour': [flt_time.hour],  # Ввод пользователя
    'fltMinute': [flt_time.minute]  # Ввод пользователя
})

combined_data = {**new_data.iloc[0].to_dict(), **transformed_data}

with mod1:
    st.write("Прогноз модели:")

    with st.spinner('Загрузка моделей и прогноз...'): # Загрузка моделей и прогноз
        for model_name in model_names:

            model_path = f'BaggingRegressors2/BaggingRegressor_{model_name}'
            model = joblib.load(model_path)

            # Прогноз для текущей модели
            prediction = model.predict([list(combined_data.values())])

            # Сохранение прогноза в словаре
            predictions_dict[model_name] = prediction

        # Вывод прогноза для каждой модели
        for model_name, prediction in predictions_dict.items():
            st.write(f'{model_name.replace("BaggingRegressor_", "")}: {float(prediction)}')
        st.success('Предсказание успешно завершено!')

with mod2:
    st.write("Прогноз модели:")

    with st.spinner('Загрузка моделей и прогноз...'):  # Загрузка моделей и прогноз
        for model_name in model_names:

            model_path = f'HistGradientBoostingRegressors/HistGradientBoostingRegressor_{model_name}'
            model = joblib.load(model_path)

            # Прогноз для текущей модели
            prediction = model.predict([list(combined_data.values())])

            # Сохранение прогноза в словаре
            predictions_dict[model_name] = prediction

        # Вывод прогноза для каждой модели
        for model_name, prediction in predictions_dict.items():
            st.write(f'{model_name.replace("HistGradientBoostingRegressor_", "")}: {float(prediction)}')
        st.success('Предсказание успешно завершено!')



