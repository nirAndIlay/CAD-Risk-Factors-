import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('merged.csv')

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

# Save the processed DataFrame to a new CSV file
df.to_csv('processed_data.csv', index=False)

print("Preprocessing complete. Processed data saved to 'processed_data.csv'.")
