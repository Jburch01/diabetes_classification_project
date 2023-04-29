import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split





def get_data():
        df = pd.read_csv('diabetes_prediction_dataset.csv')
        df.drop_duplicates(ignore_index=True, inplace=True)
        df = df.drop(df[df.age < 1].index)
        df.age = df.age.astype(int)
        df = df.rename(columns={'diabetes':'diabetic'})
        mask = df.gender == 'Other'
        df = df.drop(df[mask].index)
        bins = [0, 18, 29, 39, 49, 59, 69, 80]
        #labels = ['1-18', '19-29', '30-39', '40-49', '50-59', '60-69', "70+"]
        labels = [1, 2, 3, 4, 5, 6, 7]
        df['age_bin'] = pd.cut(df['age'], bins=bins, labels=labels)
        bmi_labels = [1, 2, 3, 4, 5, 6]
        bmi_bins= [0, 18.5, 25, 30, 35, 40, 100]
        df['bmi_class'] = pd.cut(df.bmi, bins=bmi_bins, labels=bmi_labels)
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
        
    
    
