import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split





def get_data():
        df =  pd.read_csv('diabetes.csv')
        df.columns = df.columns.str.lower()
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
        
    
    
