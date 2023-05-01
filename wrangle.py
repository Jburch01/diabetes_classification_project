import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.preprocessing import MinMaxScaler
# sklearn for modeling:
from sklearn.tree import DecisionTreeClassifier,\
export_text, \
plot_tree
from sklearn.metrics import accuracy_score, \
classification_report, \
confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

metric_df = {}

def get_data():
    '''
    grabs the data from the csv and returns as a pandas dataframe
    '''
    return pd.read_csv('diabetes_prediction_dataset.csv')

def prep_data():
        df = get_data()
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
    """
    Splits data into 3 segments and stratifies on target
    requires the dataframe and target as args
    """
    train_val, test = train_test_split(df,
                                       train_size=0.8,
                                       random_state=706,
                                       stratify=df[target])
    train, validate = train_test_split(train_val,
                                       train_size=0.7,
                                       random_state=706,
                                       stratify=train_val[target])
    return train, validate, test


def scale_data(train, val, test):
    x_cols = ['HbA1c_level']
    split = [train, val, test]
    scale_list= []
    scaler = MinMaxScaler()
    scaler.fit(train[x_cols])
    for cut in split:
        cut_copy = cut.copy()
        cut_copy[x_cols] = scaler.transform(cut_copy[x_cols])
        scale_list.append(cut_copy)

    
    return scale_list[0], scale_list[1], scale_list[2] 
        

    
def get_target_and_features(train_scale, val_scale, test_scale):    
    x_cols = ['HbA1c_level', 'age_bin', 'bmi_class']
    y_cols = 'diabetic'

    x_train = train_scale[x_cols]
    y_train = train_scale[y_cols]

    x_val = val_scale[x_cols]
    y_val = val_scale[y_cols]

    x_test = test_scale[x_cols]
    y_test = test_scale[y_cols]
    return x_train, y_train, x_val, y_val, x_test, y_test

    
def get_distributions(df):
    # Create subplots for each column
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Iterate over each column and plot a histogram on its corresponding subplot
    for i, col in enumerate(df.columns):
        axes[i].hist(df[col], bins=5)
        axes[i].set_title(f'distribution of {col}')

    # Remove any unused subplots
    for i in range(len(df.columns), len(axes)):
        fig.delaxes(axes[i])

    # Adjust the spacing between subplots and display the plots
    plt.tight_layout()
    plt.show()
    

def get_age_vis(df):
    bins = [0, 18, 29, 39, 49, 59, 69, 80]
    labels = ['1-18', '19-29', '30-39', '40-49', '50-59', '60-69', "70+"]
    df['age_bin1'] = pd.cut(df['age'], bins=bins, labels=labels)

    # count the number of occurrences of each bin
    diabetic_count = df[df['diabetic'] == 1].groupby('age_bin1')['diabetic'].count()

    # create a bar plot
    sns.barplot(x=diabetic_count.index, y=diabetic_count)

    # add axis labels and title
    plt.xlabel('Age')
    plt.ylabel('Diabetic Count')
    plt.title('Binned ages with diabetes')

    # show the plot
    plt.show()
    df.drop(columns='age_bin1')
    
    
def test_age(df):
    a = 0.05
    observed = pd.crosstab(df['age_bin'], df['diabetic'], margins=True)
    chi2, p, _, hypothetical = stats.chi2_contingency(observed)
    if p < a:
        print(f'We can reject our null hypothesis: {p} < {a}')
    else:
        print('We have failed to reject our null hypothesis')
        
        
def get_A1c_vis(df): 
    # Create subplots for the boxplot and barplot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Create the boxplot on the first subplot
    sns.boxplot(y=df.HbA1c_level, x=df.diabetic, ax=axes[0])
    axes[0].set_title('Boxplot of HbA1c Level')

    # Create the barplot on the second subplot
    sns.barplot(x=df.HbA1c_level, y=df.diabetic, ax=axes[1])
    axes[1].set_title('Barplot of HbA1c Level')

    # Adjust the spacing between subplots and display the plots
    plt.tight_layout()
    plt.show()


    
def A1c_stattest(df):
    diabetic_A1c = df[df.diabetic == 1].HbA1c_level
    overall_A1c_mean = df.HbA1c_level.mean()
    test_results = stats.ttest_1samp(diabetic_A1c, overall_A1c_mean)
    return test_results


def get_bmi_vis(df):
    # Define the BMI class order
    bmi_class_order = [
        1, 2, 3, 4, 5, 6
    ]

    titles = [ '1: Underweight: BMI less than 18.5',
            '2: Normal weight: BMI between 18.5 and 24.9',
            '3: Overweight: BMI between 25 and 29.9',
            '4: Obesity (Class 1): BMI between 30 and 34.9',
            '5: Obesity (Class 2): BMI between 35 and 39.9',
            '6: Extreme obesity (Class 3): BMI of 40 or higher']

    # Define the diabetic status order
    diabetic_order = [0, 1]
    diabetic_title = ['diabetic_No', 'diabetic_Yes']

    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
    fig.suptitle('BMI Class Distribution Percentage by Diabetic Status', fontsize=20, fontweight='bold', y=1.02)

    # Loop through each BMI class and create a barplot in the corresponding subplot
    for i, bmi_class in enumerate(bmi_class_order):
        # Determine the row and column index for the current subplot
        row = i // 3
        col = i % 3

        # Subset the data to only include the current BMI class
        data_subset = df[df['bmi_class'] == bmi_class]

        # Calculate the counts of diabetic status for the current BMI class
        counts = data_subset['diabetic'].value_counts(normalize=True)

        # Create a barplot in the current subplot
        ax = sns.barplot(x=counts.index, y=counts.values, ax=axes[row, col], hue=diabetic_title)
        ax.set_title(titles[i], fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('Proportion')
        ax.set_ylim(0, 1)


    # Adjust the layout of the subplots
    plt.tight_layout()

    # Display the plot
    plt.show()

    
    
def get_bmistats(df):
    a = 0.05
    bmi_diabetic_values = df[df.diabetic == 1].bmi
    bmi_non_diabetic_values = df[df.diabetic == 0].bmi
    t , p = stats.ttest_ind(bmi_diabetic_values, bmi_non_diabetic_values)
    if p < a:
        print(f'We can reject our null hypothesis: {p} < {a}')
    else:
        print('We have failed to reject our null hypothesis')
        
        
def get_decisionTree_model(x_train, y_train, x_val, y_val, x_test, y_test,depth, t=0):
    """
    Returns a decision treen model with a max depth arg
    prints out the Accuracy of train and validate and the 
    classification report
    """
    clf = DecisionTreeClassifier(max_depth=depth, random_state=706)
    #class_weight='balanced'
    # fit the thing
    clf.fit(x_train, y_train)

    model_proba = clf.predict_proba(x_train)
    model_preds = clf.predict(x_train)

    model_score = clf.score(x_train, y_train)
    if t == 0:
        #classification report:
        print(
            classification_report(y_train,
                              model_preds))
        print('Accuracy of Random Tree classifier on training set: {:.2f}'
         .format(clf.score(x_train, y_train)))
        print('Accuracy of Random Tree classifier on validation set: {:.2f}'
         .format(clf.score(x_val, y_val)))
    else:
        print('Accuracy of logistic regression classifier on test set: {:.2f}'
         .format(clf.score(x_test, y_test)))

    



def get_random_forest(train, x_train, y_train, x_val, y_val,):
    """
    Runs through two for loops from range 1 - 5 each time increasing the max depth 
    and min sample leaf
    puts all of the models in a pandas data frame and sorts for the hightes valadation 
    Prints out the classification report on the best model
    """
    baseline_accuracy = round((train.diabetic == 0).mean(), 2)
    model_list = []

    for j in range (1, 15):
        for i in range(2, 15):
            rf = RandomForestClassifier(n_estimators=101 ,max_depth=i, min_samples_leaf=j, random_state=706)

            rf = rf.fit(x_train, y_train)
            train_accuracy = rf.score(x_train, y_train)
            validate_accuracy = rf.score(x_val, y_val)
            model_preds = rf.predict(x_train)

            output = {
                "min_samples_per_leaf": j,
                "max_depth": i,
                "train_accuracy": train_accuracy,
                "validate_accuracy": validate_accuracy,
                'model_preds': model_preds
            }
            model_list.append(output)
            
    df = pd.DataFrame(model_list)
    df["difference"] = df.train_accuracy - df.validate_accuracy
    df["baseline_accuracy"] = baseline_accuracy
    # df[df.validate_accuracy > df.baseline_accuracy + .05].sort_values(by=['difference'], ascending=True).head(15)
    df.sort_values(by=['validate_accuracy'], ascending=False).head(1)
    
    #classification report:
    print(classification_report(y_train, df['model_preds'][1]))
    return df.sort_values(by=['validate_accuracy'], ascending=False).head(1)



def get_logReg_model(x_train, y_train, x_val, y_val):
    """
    build a logistical regression model and prints out the accuracy on training and validation along with the classification report. 
    Must type in train_val as your data arrg to get the train val result.
    Type test if you want to test the model
    if you want a csv of the model preds and preds proba then un comment all of the stuff at the bottom
    """
    logit = LogisticRegression(random_state=706)
    logit.fit(x_train, y_train)
    y_pred = logit.predict(x_train)
    y_proba = logit.predict_proba(x_train)
    logit_val = logit.predict(x_val)
   
    print('Accuracy of Logistic Regression classifier on training set: {:.2f}'
     .format(logit.score(x_train, y_train)))
    print('Accuracy of Logistic Regression classifier on validation set: {:.2f}'
     .format(logit.score(x_val, y_val)))
    print(
    classification_report(y_train,
                      y_pred))
    # else: 
    #     print('Accuracy of logistic regression classifier on test set: {:.2f}'
    #      .format(logit.score(x_test, y_test)))

    
def get_knn(x_train, y_train, x_val, y_val):
    k = 9
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')

    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_val)

    print('Accuracy of Logistic Regression classifier on training set: {:.2f}'
     .format(knn.score(x_train, y_train)))
    print('Accuracy of Logistic Regression classifier on validation set: {:.2f}'
     .format(knn.score(x_val, y_val)))
    print(classification_report(y_val,
                              y_pred))


def compare_models(x_train, y_train, x_val, y_val, x_test, y_test):
    '''
    Runs all of the the models on train and validate and returns the results in a
    data frame 
    '''
    global metric_df
    get_baseline(y_train, y_val, s=1)
    lr(x_train, y_train, x_val, y_val, s=1)
    lassolars(x_train, y_train, x_val, y_val, s=1)
    tweedie(x_train, y_train, x_val, y_val, s=1)
    get_poly(x_train, y_train, x_val, y_val, x_test, y_test, s=1)
    
    return metric_df

