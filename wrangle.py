import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split





def get_data():
        df = pd.read_csv('diabetes_prediction_dataset.csv')
        df.drop_duplicates(ignore_index=True, inplace=True)
        df = df.drop(df[df.age < 1].index)
        df = df.rename(columns={'diabetes':'diabetic'})
        return df
    
    
def train_validate_test(df,target):
    train_val, test = train_test_split(df,
                                       train_size=0.8,
                                       random_state=706,
                                       stratify=df[target])
    train, validate = train_test_split(train_val,
                                       train_size=0.7,
                                       random_state=706,
                                       stratify=train_val[target])
    return train, validate, test
        
    
    
