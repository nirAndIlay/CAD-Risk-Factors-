import pandas as pd
from tabulate import tabulate

df_path='/home/binjaminni@mta.ac.il/thinclient_drives/Create_Df/merged.csv'
handle_null_path='/home/binjaminni@mta.ac.il/thinclient_drives/Handle_Nulls/Handle_null.xlsx'

def handle_null_values(df, instructions):
    for column, action in instructions.items():
        if action == -10:
            df = df.dropna(subset=[column])
        else:
            df[column] = df[column].fillna(action)
    return df

try:
    df = pd.read_csv(df_path)
    df_instructions = pd.read_excel(handle_null_path)
except FileNotFoundError:
    print(f"File not found. Please check the file paths.")
    exit()

instruction_dict = df_instructions.set_index('Field ID').to_dict()['NULL']

if df.isnull().values.any():
    df_processed = handle_null_values(df, instruction_dict)
else:
    print("No null values found in the DataFrame.")
    df_processed = df

# Clean the DataFrame before saving it
for col in df_processed.columns:
    df_processed[col] = df_processed[col].apply(lambda x: x.replace(',', '') if isinstance(x, str) else x)

df_processed.to_csv('Handle_Null_data.csv', index=False)