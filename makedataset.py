import os
import pandas as pd
from sklearn.model_selection import train_test_split

input_folder_path = 'data_con-30'
output_folder_path = 'data_c30_467_0'

input_files = os.listdir(input_folder_path)
output_files = os.listdir(output_folder_path)

data = pd.DataFrame({'Input': input_files, 'Output': output_files})

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)