import pandas as pd
import os

# List of field IDs to filter
handle_null_path='/home/binjaminni@mta.ac.il/thinclient_drives/Handle_Nulls/Handle_null.xlsx'
features_ids_frame = pd.read_excel(handle_null_path)
field_ids = list(features_ids_frame['Field ID'])
print("Field IDS:")
print(field_ids)

path_1 ='/home/binjaminni@mta.ac.il/biobank/ukb672220.csv'
path_2 ='/home/binjaminni@mta.ac.il/biobank/ukb673316.csv'
path_3='/home/binjaminni@mta.ac.il/biobank/ukb673540.csv'

# Function to read headers of a CSV file
def read_headers(file_path):
    return pd.read_csv(file_path, nrows=0).columns.tolist()

# Read headers of each CSV file
headers_df1 = read_headers(path_1)
headers_df2 = read_headers(path_2)
headers_df3 = read_headers(path_3)

# Identify which field IDs are in each dataframe
fields_in_df1 = [fid for fid in field_ids if fid in headers_df1]
fields_in_df2 = [fid for fid in field_ids if fid in headers_df2]
fields_in_df3 = [fid for fid in field_ids if fid in headers_df3]

# Add 'eid' to the list of columns to read (assuming 'eid' is the common participant ID column)
fields_in_df1.append('eid')
fields_in_df2.append('eid')
fields_in_df3.append('eid')

# Read only the necessary columns from each CSV file
df1 = pd.read_csv(path_1, usecols=fields_in_df1)
df2 = pd.read_csv(path_2, usecols=fields_in_df2)
df3 = pd.read_csv(path_3, usecols=fields_in_df3)

duplicate_columns = ['2443-0.0', '30010-0.0', '30120-0.0', '30020-0.0', '30080-0.0']
dataframes = [df1, df2, df3]

# Handle with duplicate columns
for duplicate_column in duplicate_columns:
    dfs_with_duplicate = [df for df in dataframes if duplicate_column in df.columns]

    if len(dfs_with_duplicate) > 1:
        for df in dfs_with_duplicate[1:]:
            df.drop(columns=duplicate_column, inplace=True)

# Merge the dataframes on 'eid'
merged_df = df1.merge(df2, on='eid', how='outer')
merged_df = merged_df.merge(df3, on='eid', how='outer')

script_dir = os.path.dirname(os.path.abspath('__file__'))

output_file_path = os.path.join(script_dir, 'merged.csv')
merged_df.to_csv(output_file_path, index=False)
print(merged_df.head())