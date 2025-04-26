import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytz
import os
import plotly.express as px
import plotly.graph_objects as go
from utils import merge_csv_files
# Noise data

folder_path = "../data/"
file_list_40 = ["csv_results_40_255439_mp-01-naamsestraat-35-maxim.csv",
               "csv_results_40_255440_mp-02-naamsestraat-57-xior.csv",
               "csv_results_40_255441_mp-03-naamsestraat-62-taste.csv",
               "csv_results_40_255442_mp-05-calvariekapel-ku-leuven.csv",
               "csv_results_40_255443_mp-06-parkstraat-2-la-filosovia.csv",
               "csv_results_40_255444_mp-07-naamsestraat-81.csv",
               "csv_results_40_255445_mp-08-kiosk-stadspark.csv",
               "csv_results_40_280324_mp08bis---vrijthof.csv",
               "csv_results_40_303910_mp-04-his-hears.csv"]

file_list_41 = ["csv_results_41_255439_mp-01-naamsestraat-35-maxim.csv",
               "csv_results_41_255440_mp-02-naamsestraat-57-xior.csv",
               "csv_results_41_255441_mp-03-naamsestraat-62-taste.csv",
               "csv_results_41_255442_mp-05-calvariekapel-ku-leuven.csv",
               "csv_results_41_255443_mp-06-parkstraat-2-la-filosovia.csv",
               "csv_results_41_255444_mp-07-naamsestraat-81.csv",
               "csv_results_41_255445_mp-08-kiosk-stadspark.csv",
               "csv_results_41_280324_mp08bis---vrijthof.csv",
               "csv_results_41_303910_mp-04-his-hears.csv"]

file_list_42 = ["csv_results_42_255439_mp-01-naamsestraat-35-maxim.csv",
               "csv_results_42_255440_mp-02-naamsestraat-57-xior.csv",
               "csv_results_42_255441_mp-03-naamsestraat-62-taste.csv",
               "csv_results_42_255442_mp-05-calvariekapel-ku-leuven.csv",
               "csv_results_42_255443_mp-06-parkstraat-2-la-filosovia.csv",
               "csv_results_42_255444_mp-07-naamsestraat-81.csv",
               "csv_results_42_255445_mp-08-kiosk-stadspark.csv",
               "csv_results_42_280324_mp08bis---vrijthof.csv",
               "csv_results_42_303910_mp-04-his-hears.csv"]

# lots of files, takes a while
file40 = merge_csv_files(folder_path + "/export_40/",file_list_40)
file41 = merge_csv_files(folder_path + "/export_41/",file_list_41)
file42 = merge_csv_files(folder_path + "/export_42/",file_list_42) #Uses the incomplete, reduced data set


file_list_meteo = ["LC_2022Q1.csv","LC_2022Q2.csv","LC_2022Q3.csv","LC_2022Q4.csv",]

# lots of files, takes a while
meteo = merge_csv_files(folder_path + "/meteodata/",file_list_meteo,delim=',')