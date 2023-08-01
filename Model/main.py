from train import *
from model import *
from train import main


def main():
     # Загружаем данные и предобрабатываем их
    data = load_data(Config.DATA_FILE_PATH)
    df = process_string_columns(data, ['Final_price', 'Hum_hours', 'Process_volume'])

    # Определяем признаки и целевую переменную (таргет)
    X = df[["Object_area", "Process_volume", "Название процесса"]]
    y_price = df["Final_price"]
    y_hours = df["Hum_hours"]

    # Создаем экземпляр модели и Создаем модель
    model = Model()
    model.create_model()

    # Создаем экземпляр для OHE
    encoder = OneHotEncoder()

    # Преобразуем значения процесса в матрицу нулей и единиц
    X_encoded = encoder.fit_transform(X[["Название процесса"]])
    X_final = pd.concat([pd.DataFrame(X_encoded.toarray()), X[["Object_area", "Process_volume"]]], axis=1)

    #Чтобы избежать ошибки с int col заранее, переназначаем тип данных на str
    X_final.columns = X_final.columns.astype(str)
    
    # Разделяем данные на обучающую и тестовую выборку
    X_train, X_test, y_train_price, y_test_price, y_train_hours, y_test_hours = train_test_split(
        X_final, y_price, y_hours, test_size=0.2, random_state=42)

    # Фитим модель
    model.fit(X_train, X_test, y_train_price, y_test_price, y_train_hours, y_test_hours)

    # Предиктим результаты
    y_pred_price = model.predict_price(X_test)
    y_pred_hours = model.predict_hours(X_test)

    # Оцениваем модель
    mae_price, mae_hours = model.evaluate(y_test_price, y_pred_price, y_test_hours, y_pred_hours)

    print("MAE финансовых затрат:", mae_price)
    print("MAE времени завершения:", mae_hours)

#____________________________________________________________________________________________________
    # Предикт для входных данных.
    input_area = input('Введите площадь объекта: ')
    input_process_volume = input('Введите объем материала: ')
    input_process_name = input('Введите название процесса: ')

    #encoding process_name
    df = pd.DataFrame({"Object_area": [input_area],
                       "Process_volume": [input_process_volume],
                       "Название процесса": [input_process_name]})
    
    encoded_data = encoder.transform(df[["Название процесса"]])    
    final_data = pd.concat([pd.DataFrame(encoded_data.toarray()), df[['Object_area', 'Process_volume']]], axis = 1)
    final_data.columns = final_data.columns.astype(str)
    
    predict_price = model.predict_price(final_data)
    predict_hours = model.predict_hours(final_data)
    
    
    result_message = print(f'''
Модель предсказывает финансовые затраты: {int(predict_price)} рублей.
Модель предсказывает затраты часов на работу: {int(predict_hours)} часов.
    ''')
    return result_message

# иф нейм равно мэйн :) 
if __name__ == "__main__":
    main()