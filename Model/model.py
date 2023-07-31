from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd

class Model:
    def __init__(self):
        self.model_price = None
        self.model_hours = None
    
    
    def create_model(self):
        self.model_price = LinearRegression()
        self.model_hours = LinearRegression()

    def fit(self, X_train: pd.DataFrame,
             X_test: pd.DataFrame,
                y_train_price: pd.DataFrame,
             y_test_price: pd.DataFrame, 
                y_train_hours: pd.DataFrame, 
             y_test_hours: pd.DataFrame):
        
        self.model_price.fit(X_train, y_train_price)
        self.model_hours.fit(X_train, y_train_hours)
    

    def predict_price(self, X_test: pd.DataFrame): return self.model_price.predict(X_test)
    
    def predict_hours(self, X_test_hours): return self.model_hours.predict(X_test_hours)

    
    def evaluate(self, y_true_price, y_pred_price, y_true_hours, y_pred_hours) -> float:
        mae_price = mean_absolute_error(y_true_price, y_pred_price)
        mae_hours = mean_absolute_error(y_true_hours, y_pred_hours)
        
        return mae_price, mae_hours