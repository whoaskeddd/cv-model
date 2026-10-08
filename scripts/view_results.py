import pandas as pd

df = pd.read_csv("runs/classify/train-3/results.csv")
df1 = pd.read_csv("runs/segmentation/yolov8n_seg-2/results.csv")

print(df, df1.info())
