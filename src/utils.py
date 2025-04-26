import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytz
import os
import plotly.express as px
import plotly.graph_objects as go

def merge_csv_files(directory_path):

    """
    Merges all CSV files in a directory into a single pandas DataFrame.

    Args:
    directory_path (str): The directory path containing the CSV files.

    Returns:
    merged_df (pandas.DataFrame): The merged pandas DataFrame of all CSV files in the directory.
    """
    dfs = []

    # loop through each file in the directory
    for file in os.listdir(directory_path):
        # check if the file is a CSV file
        if file.endswith(".csv"):
            file_path = os.path.join(directory_path, file)
            if directory_path.endswith("dataverse_file"):
                df = pd.read_csv(file_path, delimiter=",")
            else:
                df = pd.read_csv(file_path, delimiter=";")
            dfs.append(df)

    # concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)

    return merged_df


def add_object_id(hourly_df, object_ids):
    dfs = []
    for id in object_ids:
        new_df = hourly_df.copy()
        new_df["object_id"] = id
        dfs.append(new_df)

    combined_df = pd.concat(dfs)
    combined_df.reset_index(drop=True, inplace=True)
    return combined_df