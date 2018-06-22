import pandas as pd
import numpy as np
import os
import shutil

ORIGIN_DIR = r'D:\Documents\smartcity\dataset\face_a\test_a\gallery'
TARGET_DIR = r'D:\Documents\smartcity\dataset\face\test'
CSV_FILE = r'D:\Documents\smartcity\dataset\face_a\test_a_gallery.csv'


read_csv = pd.read_csv(CSV_FILE, header=None)

if not os.path.exists(TARGET_DIR):
    os.mkdir(TARGET_DIR)

for i in np.arange(len(read_csv)):
    figure_path_new = os.path.join(TARGET_DIR, read_csv[1][i].astype(np.str))
    if not os.path.exists(figure_path_new):
        os.mkdir(figure_path_new)
    figure_path_original = os.path.join(ORIGIN_DIR, read_csv[0][i])
    shutil.copy(figure_path_original, figure_path_new)

print("DONE！")