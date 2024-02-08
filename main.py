import streamlit as st
import joblib
import datetime
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
st.write("""
# Прогноз количества мест на рейсе
""")
mod1, mod2, mod3 = st.tabs(["1 Модель", "2 Модель", "3 Модель"])
# Список названий моделей
model_names = [
    'BaggingRegressor_Y',
    'BaggingRegressor_P',
    'BaggingRegressor_E',
    'BaggingRegressor_B',
    'BaggingRegressor_K',
    'BaggingRegressor_L',
    'BaggingRegressor_V',
    'BaggingRegressor_Q',
    'BaggingRegressor_X',
    'BaggingRegressor_M',
    'BaggingRegressor_J',
    'BaggingRegressor_C',
    'BaggingRegressor_D',
    'BaggingRegressor_O',
    'BaggingRegressor_H',
    'BaggingRegressor_Z',
    'BaggingRegressor_T',
    'BaggingRegressor_I',
    'BaggingRegressor_N',
    'BaggingRegressor_G',
    'BaggingRegressor_R',
    'BaggingRegressor_U'
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


with mod1:
    input_date = st.date_input("Выберите дату:", datetime.datetime(2019, 12, 31))
    transformed_data = transform_date(input_date)

    flt_time = st.time_input('Выберите время', value=datetime.time(13, 35))

    new_data = pd.DataFrame({
        'SA': [0.0],  # По умолчанию
        'AU': [4.0],  # По умолчанию
        'NS': [0.0],  # По умолчанию
        'DTD': [st.number_input('Введите количество дней до вылета рейса:', min_value=0)],  # Ввод пользователя
        'FLTNUM': [st.number_input('Введите № рейса:', min_value=0)],  # Ввод пользователя
        'fltHour': [flt_time.hour],  # Ввод пользователя
        'fltMinute': [flt_time.minute]  # Ввод пользователя
    })

    combined_data = {**new_data.iloc[0].to_dict(), **transformed_data}
    st.write("Прогноз модели:")

    with st.spinner('Загрузка моделей и прогноз...'): # Загрузка моделей и предсказания
        for model_name in model_names:

            model_path = f'models/{model_name}'
            model = joblib.load(model_path)

            # Предсказание для текущей модели
            prediction = model.predict([list(combined_data.values())])

            # Сохранение предсказания в словаре
            predictions_dict[model_name] = prediction

        # Вывод предсказаний для каждой модели
        for model_name, prediction in predictions_dict.items():
            st.write(f'{model_name.replace("BaggingRegressor_", "")}: {float(prediction)}')
        st.success('Предсказание успешно завершено!')


