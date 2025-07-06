# ders9 eksikveriler.py


# bu de rste bazen veriler içinde eksiklikler olabilir bu verilerin giderilmesi doğru  değerlendirme için daha iyi olur .

# DERS 9 EKSİK VERİLER cvs dosyasındaki eksik verileri doldurma ve işleme alma işlemleri yapılacak.
# dosyada eksik veriler nan olarak işaretlenir

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




veriler=pd.read_csv('eksikveriler.csv')
print(veriler)




# DERS 9 EKSİK VERİLER i tamamlama 


from sklearn.impute import SimpleImputer


imputer=SimpleImputer(missing_values=np.nan,strategy ='mean')
# dosyada nan olan değerleri mean yani ortalamaneyse o olarak yazar günceller 


yas=veriler.iloc[:,1:4].values
print(yas)

#
imputer=imputer.fit(yas[:,1:4])# bu kısımda fit metodu ile eksik veriler için ortalama değerleri hesaplanır

yas[:,1:4]=imputer.transform(yas[:,1:4]) #transform metodu ile eksik veriler doldurulur
print(yas) # print ile doldurulmuş veriler yazdırılır



    


    
# eksik veriler 
