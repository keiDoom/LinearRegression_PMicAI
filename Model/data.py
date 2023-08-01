import pandas as pd

def load_data(file_path: str) -> pd.DataFrame: 
    return pd.read_csv(file_path)

def process_string_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:

    try:    
        for column in columns:
            df[column] = df[column].str.replace('\xa0', '').str.replace(',', '.').astype(float)
        return df
    
    except Exception:
        return df