import pandas as pd
from tabulate import tabulate

df_path='/home/binjaminni@mta.ac.il/thinclient_drives/Create_Df/merged.csv'
handle_null_path='/home/binjaminni@mta.ac.il/thinclient_drives/Handle_Nulls/Handle_null.xlsx'

def handle_null_values(df, instructions):
    # Loop through the instruction dictionary
    for column, action in instructions.items():
        if action == -10:
            # Delete rows with null values in the specified column
            df = df.dropna(subset=[column])
        else:
            # Replace null values with the specified number
            df[column] = df[column].fillna(action)

    return df


# Example usage:
# Load the Excel file
df = pd.read_csv(df_path, nrows = 50)

# Convert the '53-0.0' column to datetime format
df['53-0.0'] = pd.to_datetime(df['53-0.0'])

# Define the instruction dictionary
# Load the Excel file
df_instructions = pd.read_excel(handle_null_path)




# Convert the DataFrame to a dictionary
instruction_dict = df_instructions.set_index('Field ID').to_dict()['NULL']



# Apply the function
df_processed = handle_null_values(df, instruction_dict)

df_processed['53-0.0'] = pd.to_datetime(df_processed['53-0.0'], format='%Y-%m-%d')
# Extract year from '53-0.0'
df_processed['activity_year'] = df_processed['53-0.0'].dt.year
df_processed['Age when attended to assessment center'] = df_processed['activity_year'] - df_processed['34-0.0']


print(df_processed.head(1)['23407-0.0'])

