# ders7_verileri_yukleme.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




veriler=pd.read_csv('veriler.csv')
print(veriler)

boy=veriler[['boy']]  # boy verisini al dataframe olarak
print(boy)
kilo=veriler[['kilo']] # kilo verisini al dataframe olarak
print(kilo)
boykilo=veriler[["boy","kilo"]] #dataframe olarak boy kilo stununu çekti
print(boykilo )

x=10
print(x)



# DERS 8 EKSİK VERİLER 

class insan:
    boy=180
    
    def kosmak(self,b):
        return b + 10
    




ali=insan() # ali nesnesi tanımlama 

print(ali.boy) #ali özellikleri tanımlama 

print(ali.kosmak(80)) # fonku çağırma


liste=[1,2,3] #liste tanımlaması 




    