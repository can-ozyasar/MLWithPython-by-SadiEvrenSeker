import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('odev_tenis_Veri.csv')
#pd.read_csv("veriler.csv")

windy = veriler.iloc[:,3:4].values
print(windy)