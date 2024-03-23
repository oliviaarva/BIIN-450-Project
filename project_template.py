import pandas as pd

# Replace 'file_path.csv' with the path to your CSV file
file_path = 'CD_data.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Print the first 5 rows of the DataFrame
print(df.head(5))


