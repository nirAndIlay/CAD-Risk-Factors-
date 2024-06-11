import pandas as pd
import os 

# List of field IDs to filter
features_ids_frame = pd.read_excel('Hadle_null.xlsx')
field_ids = list(features_ids_frame['Field ID'])
print("Field IDS:")
print(field_ids)


# Function to read headers of a CSV file
def read_headers(file_path):
    return pd.read_csv(file_path, nrows=0).columns.tolist()

# Read headers of each CSV file
headers_df1 = read_headers('/home/ilayed@mta.ac.il/biobank/ukb672220.csv')
headers_df2 = read_headers('/home/ilayed@mta.ac.il/biobank/ukb673316.csv')
headers_df3 = read_headers('/home/ilayed@mta.ac.il/biobank/ukb673540.csv')



# Identify which field IDs are in each dataframe
fields_in_df1 = [fid for fid in field_ids if fid in headers_df1]
fields_in_df2 = [fid for fid in field_ids if fid in headers_df2]
fields_in_df3 = [fid for fid in field_ids if fid in headers_df3]

# Add 'eid' to the list of columns to read (assuming 'eid' is the common participant ID column)
fields_in_df1.append('eid')
fields_in_df2.append('eid')
fields_in_df3.append('eid')

# Read only the necessary columns from each CSV file
df1 = pd.read_csv('/home/ilayed@mta.ac.il/biobank/ukb672220.csv', usecols=fields_in_df1)
df2 = pd.read_csv('/home/ilayed@mta.ac.il/biobank/ukb673316.csv', usecols=fields_in_df2)
df3 = pd.read_csv('/home/ilayed@mta.ac.il/biobank/ukb673540.csv', usecols=fields_in_df3)

# Merge the dataframes on 'eid'
merged_df = df1.merge(df2, on='eid', how='outer')
merged_df = merged_df.merge(df3, on='eid', how='outer')



script_dir = os.path.dirname(os.path.abspath('__file__'))

output_file_path = os.path.join(script_dir, 'merged.csv')
merged_df.to_csv(output_file_path, index=False)
# Display the first few rows of the merged dataframe
print(merged_df.head())

