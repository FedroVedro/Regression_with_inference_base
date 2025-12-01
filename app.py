import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Загружаем модель
@st.cache_resource
def load_model():
    with open('model.pickle', 'rb') as f:
        model_data = pickle.load(f)
    return model_data

# Загружаем данные для EDA
@st.cache_data
def load_data():
    df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    
    # Быстрая предобработка для графиков
    df_train['mileage'] = pd.to_numeric(df_train['mileage'].str.split(' ').str[0], errors='coerce')
    df_train['engine'] = pd.to_numeric(df_train['engine'].str.split(' ').str[0], errors='coerce')
    df_train['max_power'] = pd.to_numeric(df_train['max_power'].str.split(' ').str[0], errors='coerce')
    
    return df_train

# Загружаем всё
model_data = load_model()
df = load_data()

st.title('🚗 Предсказание цены автомобиля')
st.markdown('---')

# Создаем табы
tab1, tab2, tab3 = st.tabs(['📊 EDA', '🔮 Предсказание', '📈 Веса модели'])

# EDA 
with tab1:
    st.header('Анализ данных')
    st.write('Тут можно посмотреть основные графики и гистограммы по нашим данным')
    
    #  Распределение цен
    st.subheader('Распределение цен на автомобили')
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.hist(df['selling_price'], bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Цена')
    ax1.set_ylabel('Количество')
    ax1.set_title('Гистограмма цен')
    st.pyplot(fig1)
    
    # Boxplot по типу топлива
    st.subheader('Распределение цен по типу топлива')
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    df.boxplot(column='selling_price', by='fuel', ax=ax2)
    ax2.set_xlabel('Тип топлива')
    ax2.set_ylabel('Цена')
    ax2.set_title('Цена в зависимости от типа топлива')
    plt.suptitle('')
    st.pyplot(fig2)
    
    # Зависимость цены от мощности
    st.subheader('Зависимость цены от мощности')
    fig3 = px.scatter(df.sample(1000), x='max_power', y='selling_price', 
                     opacity=0.5, title='Цена vs Мощность',
                     labels={'max_power': 'Мощность (bhp)', 'selling_price': 'Цена'})
    st.plotly_chart(fig3)
    
    # Корреляционная матрица
    st.subheader('Корреляционная матрица')
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr = df[numeric_cols].corr()
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax4, vmin=-1, vmax=1)
    ax4.set_title('Корреляция признаков')
    st.pyplot(fig4)

# Предсказание
with tab2:
    st.header('Предсказание цены автомобиля')
    st.write('Введите данные об автомобиле или загрузите CSV файл')
    
    # Выбор способа ввода
    input_method = st.radio('Выберите способ ввода данных:', 
                            ['Ручной ввод', 'Загрузка CSV'])
    
    if input_method == 'Ручной ввод':
        st.subheader('Введите характеристики автомобиля:')
        
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.number_input('Год выпуска', min_value=1980, max_value=2024, value=2015)
            km_driven = st.number_input('Пробег (км)', min_value=0, max_value=1000000, value=50000)
            mileage = st.number_input('Расход топлива (kmpl)', min_value=0.0, max_value=50.0, value=19.0)
            engine = st.number_input('Объем двигателя (CC)', min_value=500, max_value=5000, value=1200)
            max_power = st.number_input('Мощность (bhp)', min_value=30.0, max_value=500.0, value=80.0)
        
        with col2:
            fuel = st.selectbox('Тип топлива', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
            seller_type = st.selectbox('Тип продавца', ['Individual', 'Dealer', 'Trustmark Dealer'])
            transmission = st.selectbox('Коробка передач', ['Manual', 'Automatic'])
            owner = st.selectbox('Владелец', ['First Owner', 'Second Owner', 'Third Owner', 
                                              'Fourth & Above Owner', 'Test Drive Car'])
            seats = st.number_input('Количество мест', min_value=2, max_value=14, value=5)
        
        if st.button('Предсказать цену', type='primary'):
            # Создаем датафрейм с введенными данными
            input_data = pd.DataFrame({
                'year': [year],
                'km_driven': [km_driven],
                'mileage': [mileage],
                'engine': [engine],
                'max_power': [max_power],
                'fuel': [fuel],
                'seller_type': [seller_type],
                'transmission': [transmission],
                'owner': [owner],
                'seats': [seats]
            })
            
            # Предобработка
            preprocessor = model_data['preprocessor']
            model = model_data['model']
            
            # Трансформация
            input_transformed = preprocessor.transform(input_data)
            
            # Предсказание
            prediction = model.predict(input_transformed)[0]
            
            st.success(f'### Предсказанная цена: ₹{prediction:,.2f}')
            st.info(f'Это примерно **${prediction/80:,.2f}** долларов (по курсу ~80 рупий за доллар)')
    
    else:  # Загрузка CSV
        st.subheader('Загрузите CSV файл с данными')
        st.write('Файл должен содержать столбцы: year, km_driven, mileage, engine, max_power, fuel, seller_type, transmission, owner, seats')
        
        uploaded_file = st.file_uploader('Выберите CSV файл', type=['csv'])
        
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.write('Загруженные данные:')
                st.dataframe(input_df)
                
                if st.button('Предсказать цены', type='primary'):
                    # Предобработка
                    preprocessor = model_data['preprocessor']
                    model = model_data['model']
                    
                    # Трансформация
                    input_transformed = preprocessor.transform(input_df)
                    
                    # Предсказание
                    predictions = model.predict(input_transformed)
                    
                    # Добавляем предсказания в датафрейм
                    result_df = input_df.copy()
                    result_df['predicted_price'] = predictions
                    
                    st.success('Предсказания готовы!')
                    st.dataframe(result_df)
                    
                    # Можно скачать результат
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label='Скачать результаты',
                        data=csv,
                        file_name='predictions.csv',
                        mime='text/csv'
                    )
            
            except Exception as e:
                st.error(f'Ошибка при обработке файла: {e}')
                st.write('Убедитесь, что файл содержит все необходимые столбцы')

# Веса модели 
with tab3:
    st.header('Веса обученной модели')
    st.write('Здесь можно посмотреть, какие признаки модель считает важными')
    
    # Получаем веса
    model = model_data['model']
    feature_names = model_data['feature_names']
    coefficients = model.coef_
    
    # Создаем датафрейм с весами
    coef_df = pd.DataFrame({
        'Признак': feature_names,
        'Вес': coefficients
    })
    
    # Сортируем по абсолютному значению
    coef_df['Абсолютный вес'] = coef_df['Вес'].abs()
    coef_df = coef_df.sort_values('Абсолютный вес', ascending=False)
    
    # График весов
    st.subheader('Важность признаков (по модулю весов)')
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    colors = ['green' if x > 0 else 'red' for x in coef_df['Вес'][:15]]
    ax5.barh(range(15), coef_df['Абсолютный вес'][:15], color=colors, alpha=0.7)
    ax5.set_yticks(range(15))
    ax5.set_yticklabels(coef_df['Признак'][:15])
    ax5.set_xlabel('Абсолютное значение веса')
    ax5.set_title('Топ-15 самых важных признаков')
    ax5.invert_yaxis()
    st.pyplot(fig5)
    
    st.write('🟢 **Зеленый** = положительное влияние на цену')
    st.write('🔴 **Красный** = отрицательное влияние на цену')
    
    # Таблица с весами
    st.subheader('Таблица весов всех признаков')
    st.dataframe(coef_df[['Признак', 'Вес']], height=400)
    
    # Дополнительная информация
    st.info(f'''
    **Информация о модели:**
    - Алгоритм: Ridge Regression
    - Параметр регуляризации (alpha): {model_data['best_params']['alpha']:.4f}
    - R² на кросс-валидации: {model_data['best_score']:.4f}
    - Количество признаков: {len(feature_names)}
    ''')


st.markdown('---')
st.markdown('Сделано для ДЗ №1 по ML')

