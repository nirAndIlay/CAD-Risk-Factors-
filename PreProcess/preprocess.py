import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# After Handle with null values

df_path='/home/binjaminni@mta.ac.il/thinclient_drives/Handle_Nulls/Handle_Null_data.csv'

# Load the dataset
df = pd.read_csv(df_path)

# 1. Calculate age from year of birth and date of attending assessment centre
df['Date of attending assessment centre'] = pd.to_datetime(df['53-0.0'])
df['Year of birth'] = df['34-0.0']
df['Age'] = df['Date of attending assessment centre'].dt.year - df['Year of birth']


# 2. Delete samples with specific values in Ethnicity
df = df[~df['21000-0.0'].isin([-3, -1, 6])]
# 2.1 Only relating to the main categoris of Enthnicity, without the sub-categories
df['21000-0.0'] = df['21000-0.0'].astype(str).str[0]
df['21000-0.0'] = df['21000-0.0'].astype(int)

# 3. Delete samples with specific values in smoking status
df = df[~df['20116-0.0'].isin([-1, -3])]

# 4. Delete samples with specific values in alcohol consumption frequency
df = df[~df['1558-0.0'].isin([-1,-3])]

# 5. Change values in physical activity
df['884-0.0'] = df['884-0.0'].replace({-1: 0, 3: 0})

# 6 Change values in Father history of cardiovascular diseases to binary values (1=1 other=0) -----
df['20107-0.0'] = df['20107-0.0'].replace({14: 0, 13:0 ,12:0 ,11:0 ,10:0 ,9:0, 8:0,7:0,6:0,5:0,4:0,3:0,2:1,1:1,-11:0, -13:0, -17:0, -21:0, -23:0, -27:0})

# 7 Change values in Mother history of cardiovascular diseases to binary values (1=1 other=0) -----
df['20110-0.0'] = df['20110-0.0'].replace({14: 0, 13:0 ,12:0 ,11:0 ,10:0 ,9:0, 8:0,7:0,6:0,5:0,4:0,3:0,2:1,1:1,-11:0, -13:0, -17:0, -21:0, -23:0, -27:0})

# 8. Delete samples with specific values in diabetes diagnosis
df = df[~df['2443-0.0'].isin([-3, -1])]


# 9. Delete samples with specific values in sleep duration
df = df[~df['1160-0.0'].isin([-3, -1])]

# 10. Delete samples with specific values in handedness
df = df[~df['1707-0.0'].isin([-3])]

# 11. Handle usual walking pace, deleting samples with ('None of the above', and 'Prefer not to answer)
df = df[~df['924-0.0'].isin([-3, -7])]


# 12. Convert types of physical activity in last 4 weeks to binary if x in [-3,-4,1,7] so x=0 else if in [2,3,5,6] so x=1
df['6164-0.0'] = df['6164-0.0'].replace({-3: 0, -4: 0, 1: 0, 7: 0, 2: 1, 3: 1, 5: 1, 6: 1})

# 13. Change values for field IDs 1289, 1498, 1528
fields_to_change = ['1289-0.0', '1498-0.0', '1528-0.0']
for field in fields_to_change:
    df[field] = df[field].replace({-10: 0, -3: 2, 1: 2})

# 14. Delete samples with specific values for field IDs 1329, 1369, 1548
fields_to_delete = ['1329-0.0', '1369-0.0', '1548-0.0']
for field in fields_to_delete:
    df = df[~df[field].isin([-1, -3])]


# 15 Change values in Bread intake
df['1438-0.0'] = df[field].replace({-3: 1})

# 16 Change values in Fresh fruit intake
df['1309-0.0'] = df[field].replace({-10: 0, -3: 2.29267, 1: 2.29267})

# 17 Change values in Salt added to food
df['1478-0.0'] = df[field].replace({-10: 0, -3: 1, 1: 0.5})

# 18 change Cheese intake to binary
df['1408-0.0'] = df[field].replace({-10: 0, -3: 0, -1: 0, 2: 0, 1:0, 3:1,4:1,5:1})

# 19 change Tea intake to binary
df['1488-0.0'] = df[field].replace({-10: 0, -3: 3.48432, 1: 3.48432})

# 20 change Average monthly red wine intake
df['4407-0.0'] = df[field].replace({-10: 0, -3: 2, 1: 2})

# 21.Arrange the value 131306-0.0 as our target value as a binary value (0,1)
# Define the special dates
special_dates = ['1900-01-01', '1909-09-09', '2037-07-07']
# Define a function that checks if a date is in the special group
def convert_date_to_binary(date):
    if date in special_dates:
        return 0
    else:
        return 1



# Apply the function to the target column
df['tag'] = df['131306-0.0']
df['tag']=df['tag'].apply(convert_date_to_binary)

# Filter the columns by columns [13106-0.0] < [53-0.0] date and tag == 1
df['131306-0.0'] = pd.to_datetime(df['131306-0.0'])
df = df[((df['131306-0.0'] < df['Date of attending assessment centre']) & (df['tag'] == 1)) | (df['tag'] == 0)]

# drop the columns 'Date of attending assessment centre' and 'Year of birth' after calculating age
df.drop(columns=['Date of attending assessment centre', 'Year of birth','53-0.0', '34-0.0','131306-0.0'], inplace=True)

# 16. Choose randomly 300,000 samples from the dataset with tag==0, and keep all samples with tag==1
df_tag_0 = df[df['tag'] == 0].sample(n=300000, random_state=42)
df_tag_1 = df[df['tag'] == 1]

# Merge the two DataFrames in random order to shuffle the data
df = pd.concat([df_tag_0, df_tag_1])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# Normalize all df columns except the target and eid and the categorical columns
scaler = MinMaxScaler()
columns_to_normalize = df.columns.difference(['eid','tag','21000-0.0','20116-0.0','1558-0.0','1707-0.0','924-0.0','6164-0.0','1329-0.0','1369-0.0','1548-0.0'])
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])


#One-Hot-Encoding for categorial veraibles
one_hot_encoded = pd.get_dummies(df, columns=[
    '21000-0.0',
    '20116-0.0',
    '1558-0.0',
    '1707-0.0',
    '924-0.0',
    '1329-0.0',
    '1369-0.0',
    '1548-0.0',
], prefix=[
    'Ethnicity',
    'Smoking',
    'Alcohol',
    'Handedness',
    'Usual Walking Pace',
    'Oily fish intake',
    'Beef intake',
    'Variation in diet'
])

cols_to_convert = ['Ethnicity_1','Ethnicity_2','Ethnicity_3','Ethnicity_4','Ethnicity_5',
'Smoking_0.0','Smoking_1.0','Smoking_2.0',
'Alcohol_1.0','Alcohol_2.0','Alcohol_3.0','Alcohol_4.0','Alcohol_5.0','Alcohol_6.0',
'Handedness_1.0','Handedness_2.0','Handedness_3.0',
'Usual Walking Pace_1.0','Usual Walking Pace_2.0','Usual Walking Pace_3.0',
'Oily fish intake_0.0','Oily fish intake_1.0','Oily fish intake_2.0','Oily fish intake_3.0','Oily fish intake_4.0','Oily fish intake_5.0',
'Beef intake_0.0','Beef intake_1.0','Beef intake_2.0','Beef intake_3.0','Beef intake_4.0','Beef intake_5.0',
'Variation in diet_1.0','Variation in diet_2.0','Variation in diet_3.0']

one_hot_encoded[cols_to_convert] = one_hot_encoded[cols_to_convert].astype(int)

# Save the processed DataFrame to a new CSV file
one_hot_encoded.to_csv('processed_data.csv', index=False)

print("Preprocessing complete. Processed data saved to 'processed_data.csv'.")