from flask import Flask, request, jsonify
from model import Model
from data import load_data, process_string_columns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from config import Config
import pandas as pd
from flask_cors import CORS



app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

# Создаем экземпляр модели
model = Model()
model.create_model()

# Создаем экземпляр для OHxE
encoder = OneHotEncoder()



@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type, Access-Control-Allow-Headers'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

def add_header(response):

    response.headers['Cache-Control'] = 'no-store'

    return response

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Получите данные ввода непосредственно из запроса
    input_data = pd.DataFrame(data=[data], columns=['Object_area', 'Название процесса'])

    # Загружаем данные и предобрабатываем их
    df = load_data(Config.DATA_FILE_PATH)
    df = process_string_columns(df, ['Final_price', 'Hum_hours'])

    # Определяем признаки и целевую переменную (таргет)
    X = df[["Object_area", "Название процесса"]]
    
    # Обучаем кодировщик OneHotEncoder
    encoder.fit(X[["Название процесса"]])

    # Преобразуем значения процесса в матрицу нулей и единиц
    X_encoded = encoder.transform(X[["Название процесса"]])
    X_final = pd.concat([pd.DataFrame(X_encoded.toarray()), X[["Object_area"]]], axis=1)
    X_final.columns = X_final.columns.astype(str)

    # Разделяем данные на обучающую и тестовую выборку
    X_train, X_test, y_train_price, y_test_price, y_train_hours, y_test_hours = train_test_split(
        X_final, df["Final_price"], df["Hum_hours"], test_size=0.2, random_state=42)

    # Фитим модель
    model.fit(X_train, X_test, y_train_price, y_test_price, y_train_hours, y_test_hours)

    # Получаем данные для предсказания из запроса
    input_data = process_string_columns(input_data, ['Object_area'])
    input_data_encoded = encoder.transform(input_data[["Название процесса"]])
    
    final_data = pd.concat([pd.DataFrame(input_data_encoded.toarray()), input_data[['Object_area']]], axis=1)
    final_data.columns = final_data.columns.astype(str)

    # Предиктим результаты.
    predicted_price = model.predict_price(final_data)
    predicted_hours = model.predict_hours(final_data)

    # Возвращаем результат в формате JSON.
    result = {
    "predicted_hours": int(predicted_hours[0]),
    "predicted_price": int(predicted_price[0])
}

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='194-58-98-29.cloudvps.regruhosting.ru', debug=True, ssl_context='adhoc')