import os
import pandas as pd
from pathlib import Path

def load_metadata(root_data='./DATA'):

    # Read metadata file
    metadata_file_path = os.path.join(root_data, 'metadata/UrbanSound8K.csv')
    df = pd.read_csv(metadata_file_path)
    print(df.head())

    # Construct file path by concatenating fold and file name
    df['relative_path'] = 'audio/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)

    # Take relevant columns
    # df = df[['relative_path', 'classID']]
    return df
