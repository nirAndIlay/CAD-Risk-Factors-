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


# 3. Delete samples with specific values in Ethnicity
df = df[~df['21000-0.0'].isin([-3, -1, 6])]

# 4. Delete samples with specific values in smoking status
df = df[~df['20116-0.0'].isin([-1, -3])]

# 5. Delete samples with specific values in alcohol consumption frequency
df = df[~df['1558-0.0'].isin([-3])]

# 6. Change values in physical activity
df['884-0.0'] = df['884-0.0'].replace({-1: 0, 3: 0})

# 7. Delete samples with specific values in diabetes diagnosis
df = df[~df['2443-0.0'].isin([-3, -1])]

# 8. Normalize specific fields
scaler = MinMaxScaler()
df[['30010-0.0', '30120-0.0', '30020-0.0']] = scaler.fit_transform(df[['30010-0.0', '30120-0.0', '30020-0.0']])

# 9. Delete samples with specific values in sleep duration
df = df[~df['1160-0.0'].isin([-3, -1])]

# 10. Delete samples with specific values in handedness
df = df[~df['1707-0.0'].isin([-3])]

# 11. Handle usual walking pace
df = df[~df['924-0.0'].isin([-3])]
df['924-0.0'] = df['924-0.0'].replace({-7: -2})

# 12. Convert types of physical activity in last 4 weeks to binary
df['6164-0.0'] = df['6164-0.0'].apply(lambda x: 1 if x in [-3, -4, 1, 7] else 0 if x in [2, 3, 5] else x)

# 13. Change values for field IDs 1289, 1498, 1528
fields_to_change = ['1289-0.0', '1498-0.0', '1528-0.0']
for field in fields_to_change:
    df[field] = df[field].replace({-10: 0, -3: 2, 1: 2})

# 14. Delete samples with specific values for field IDs 1329, 1369, 1548
fields_to_delete = ['1329-0.0', '1369-0.0', '1548-0.0']
for field in fields_to_delete:
    df = df[~df[field].isin([-1, -3])]

# 15.Arrange the value 131306-0.0 as our target value as a binary value (0,1)
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



# Save the processed DataFrame to a new CSV file
df.to_csv('processed_data.csv', index=False)

print("Preprocessing complete. Processed data saved to 'processed_data.csv'.")
