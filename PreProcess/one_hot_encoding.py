import pandas as pd 

df = pd.read_csv('processed_data.csv')


# One-Hot-Encoding for categorial veraibles 
one_hot_encoded = pd.get_dummies(df, columns=[
    '21000-0.0', 
    '20116-0.0', 
    '1558-0.0', 
    '20107-0.0', 
    '20110-0.0', 
    '1707-0.0',
    '924-0.0',
    '6164-0.0',
    '1329-0.0',
    '1369-0.0',
    '1548-0.0',
], prefix=[
    'Ethnicity', 
    'Smoking', 
    'Alcohol', 
    'Father_CVD', 
    'Mother_CVD', 
    'Handedness',
    'Usual Walking Pace',
    'Types of physical activity in last 4 weeks',
    'Oily fish intake',
    'Beef intake',
    'Variation in diet'
])

df.drop(columns=[
    '21000-0.0', 
    '20116-0.0', 
    '1558-0.0', 
    '20107-0.0', 
    '20110-0.0', 
    '1707-0.0',
    '924-0.0',
    '6164-0.0',
    '1329-0.0',
    '1369-0.0',
    '1548-0.0',
],inplace=True)

df = pd.concat([df, one_hot_encoded], axis=1)


# Save the processed DataFrame to a new CSV file
df.to_csv('processed_data_with_dummies.csv', index=False)
