from flask import Flask, request, jsonify
from model import Model
from data import load_data, process_string_columns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from config import Config
import pandas as pd

app = Flask(__name__)

# Создаем экземпляр модели
model = Model()
model.create_model()

# Создаем экземпляр для OHE
encoder = OneHotEncoder()


@app.route('/predict', methods=['POST'])

def predict():
    data = request.get_json()

    # Загружаем данные и предобрабатываем их
    df = load_data(Config.DATA_FILE_PATH)
    df = process_string_columns(df, ['Final_price', 'Hum_hours', 'Process_volume'])

    # Определяем признаки и целевую переменную (таргет)
    X = df[["Object_area", "Process_volume", "Название процесса"]]
    
    # Обучаем кодировщик OneHotEncoder
    encoder.fit(X[["Название процесса"]])

    # Преобразуем значения процесса в матрицу нулей и единиц
    X_encoded = encoder.transform(X[["Название процесса"]])
    X_final = pd.concat([pd.DataFrame(X_encoded.toarray()), X[["Object_area", "Process_volume"]]], axis=1)
    X_final.columns = X_final.columns.astype(str)

    # Разделяем данные на обучающую и тестовую выборку
    X_train, X_test, y_train_price, y_test_price, y_train_hours, y_test_hours = train_test_split(
        X_final, df["Final_price"], df["Hum_hours"], test_size=0.2, random_state=42)

    # Фитим модель
    model.fit(X_train, X_test, y_train_price, y_test_price, y_train_hours, y_test_hours)

    # Получаем данные для предсказания из запроса
    input_data = pd.DataFrame(data['input_data'])
    input_data = process_string_columns(input_data, ['Object_area', 'Process_volume'])
    input_data_encoded = encoder.transform(input_data[["Название процесса"]])
    
    final_data = pd.concat([pd.DataFrame(input_data_encoded.toarray()), input_data[['Object_area', 'Process_volume']]], axis=1)
    final_data.columns = final_data.columns.astype(str)

    # Предиктим результаты
    predicted_price = model.predict_price(final_data)
    predicted_hours = model.predict_hours(final_data)

    # Возвращаем результат в формате JSON
    result = {
        'predicted_price': predicted_price.tolist(),
        'predicted_hours': predicted_hours.tolist()
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug = True)