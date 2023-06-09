# Predicting Diabetes 

---
## Executive Summary
- #### Project Goal
    - Discover key drives that have a high indication of diabetes.
    - Delvelope classification ML models for detecting diabetes in a patient
- #### Key Findings
    - Age, A1c, and BMI had the biggest impact on detecting diabetes
    - As age increases do do the number of diabetic cases
    - As A1c gets out of  range(5.7 - 6.2) more diabetic cases there are.
- #### Next Steps
    - Moving forward with this project I would drop this dataset and look to acquire a better dataset. Also I would research ever column/feature more thoroughly.
## Project Description


Diabetes is a chronic medical condition characterized by high levels of glucose (sugar) in the blood. The hormone insulin, produced by the pancreas, regulates the amount of glucose in the bloodstream. In people with diabetes, the body either doesn't produce enough insulin or doesn't use it effectively, causing glucose to build up in the blood instead of being transported into cells to be used as energy. The aim of this project is to use classification models to effectively detect diabetes using a variety of different features. Early detection is crucial for individuals with diabetes to better manage the condition. This project is a proof of concept, and I plan to further it by exploring different and more complex data sets to identify different drivers and create a more accurate detection model.



## Project Goal
--- 
- Discover key drives that have a high indication of diabetes.
- Delvelope classification ML models for detecting diabetes in a patient




## Initial Thoughts
---
Diabetes is an ever growing issue that having an effective detection system could not only potentially catch early on but give the patient a better chance at getting it under control.



## Planning
---
- ### Acquire data 
    - Acquired data from kaggle 
    - 100,000 rows
    - 9 columns
    - link can be found below in steps to reproduce
- ### Prep/clean the data 
    - Split data into train, validate, and test
- ### Explore the data
    - #### Feature Engineering
        - Grouping ages into bins represented as an encoded number:
            - 1 -  18 : 1
            - 19 - 29 : 2
            - 30 - 39 : 3
            - 40 - 49 : 4
            - 50 - 59 : 5
            - 60 - 69 : 6
            - 70+ : 7
        - Grouping bmi scores in to labels classified by the World Health Organization as:
            - Underweight: BMI < 18.5 kg/m²(1)
            - Normal weight: BMI 18.5-24.9 kg/m²(2)
            - Overweight: BMI 25-29.9 kg/m²(3)
            - Obesity class I: BMI 30-34.9 kg/m²(4)
            - Obesity class II: BMI 35-39.9 kg/m²(5)
            - Obesity class III: BMI ≥ 40 kg/m²(6)
    - Discover potentil drivers 
    - Create hypothesis driver correlation
    - Preform Statistical Test on drivers
- ### Create Models for detecting patients with and without diabetes
    - Use models on train and validate data
    - Measure Models effectiveness on train and validate
    - Select best performing model for test
- ### Draw Conclusions 


<!DOCTYPE html>
<html>
  <head>
    <style>
      table, th, td {
        border: 1px solid black;
        padding: 5px;
      }
    </style>
  </head>
  <body>
    <h2>Diabetes Data Dictionary</h2>
    <table>
      <tr>
        <th>Feature</th>
        <th>Data Type</th>
        <th>Description</th>
      </tr>
      <tr>
        <td>gender</td>
        <td>Categorical</td>
        <td>The gender of the patient (Male/Female)</td>
      </tr>
      <tr>
        <td>age</td>
        <td>Numeric</td>
        <td>The age of the patient in years</td>
      </tr>
      <tr>
        <td>hypertension</td>
        <td>Categorical</td>
        <td>Whether or not the patient has hypertension (Yes "1" /No "0")</td>
      </tr>
      <tr>
        <td>heart_disease</td>
        <td>Categorical</td>
        <td>Whether or not the patient has a history of heart disease (Yes "1" /No "0")</td>
      </tr>
      <tr>
        <td>smoking_history</td>
        <td>Categorical</td>
        <td>Whether or not the patient has a history of smoking (Yes "1" /No "0")</td>
      </tr>
      <tr>
        <td>bmi</td>
        <td>Numeric</td>
        <td>The body mass index of the patient, calculated as weight in kilograms divided by height in meters squared</td>
      </tr>
      <tr>
        <td>HbA1c_level</td>
        <td>Numeric</td>
        <td>The level of HbA1c, a measure of average blood glucose levels over the past 2-3 months</td>
      </tr>
      <tr>
        <td>blood_glucose_level</td>
        <td>Numeric</td>
        <td>The current level of glucose in the patient's blood, measured in milligrams per deciliter (mg/dL)</td>
      </tr>
      <tr>
        <td>diabetic</td>
        <td>Categorical</td>
        <td>Whether or not the patient has diabetes (Yes "1" /No "0")</td>
      </tr>
    </table>
  </body>
</html>


## Steps to Reproduce 
- Clone repo
- Accqire data from https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
- Run notebook

## Takeaways and Conclusions
- Models can  affectivly detected diabetes by HbA1c_level, age, bmi alone but could be better.
- For Blood Glucose (BG) level I decided not to use it because of the lack of context behind the number.
    - Did the patient just eat and/or what did they eat?
    - Was this after an eight hour fasting (sleep)? 
    - Was this number the last check from a two hour glucose tollerance test? 
- Without having the  answer from questions like those make this column pretty irrelivant.
- Will note that using the BG level feature will increase the model's accuracy but I deduce that's becasue the dataset is skewed towards using that number.
- Hindsight is 20/20
## Next Steps
- Moving forward with this project I would drop this dataset and look to acquire a better dataset. Also I would research ever column/feature more thoroughly. 
 