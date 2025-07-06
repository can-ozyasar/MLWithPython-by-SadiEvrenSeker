# ders10 kategorik_veriler.py


# Bu derste kategorik veriler üzerinde işlem yapacağız. Kategorik veriler, # belirli kategorilere veya sınıflara ayrılabilen verilerdir. Örneğin, cinsiyet, renk, şehir gibi veriler kategorik verilere örnek olarak verilebilir.
# Kategorik veriler genellikle sayısal olmayan değerler içerir ve bu nedenle # makine öğrenimi algoritmaları tarafından doğrudan işlenemez. Bu nedenle, kategorik verileri sayısal verilere dönüştürmek için etiket kodlama (label encoding) veya tek sıcaklık kodlama (one-hot encoding) gibi teknikler kullanılır.
# Bu derste, kategorik verileri nasıl işleyebileceğimizi ve bu verileri # makine öğrenimi modellerinde nasıl kullanabileceğimizi öğreneceğiz. Ayrıca, # eksik verileri nasıl doldurabileceğimizi de göreceğiz.
# Kategorik veriler üzerinde işlem yaparken, verilerin doğru bir şekilde # kodlanması ve eksik verilerin uygun bir şekilde doldurulması önemlidir.

#ÖZETLE kategorik verileri sayısal verilere dönüştürme işlemi yaptık . 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




veriler=pd.read_csv('kategorikVeriler.csv')
print(veriler) #  tüm verileri ekrana yazdırırız




# DERS 10 kategorik veriler i doğru sınıflandırma 


from sklearn.impute import SimpleImputer


imputer=SimpleImputer(missing_values=np.nan,strategy ='mean')
 #dosyada nan olan değerleri mean yani ortalamaneyse o olarak yazar günceller 


yas=veriler.iloc[:,1:4].values
# print(yas)


ulke =veriler.iloc[:,0:1].values
print(ulke) #ülke kolonunu alırız

from sklearn import preprocessing

le=preprocessing.LabelEncoder()


ulke[:,0] = le.fit_transform(veriler.iloc[:,0]) # ulke kolonundaki verileri etiket kodlama ile sayısal verilere dönüştürürüz
# fit_transform ile hem fit hem de transform işlemini yaparız 

print(ulke)
# [1]
#  [1]
#  [1]
#  [1]
#  [1]
#  [1]
#  [1]
#  [1]
#  [1]
#  [2]
#  [2]
#  [2]
#  [2]
#  [2]
#  [2]
#  [0]
#  [0]
#  [0]
#  [0] bu şekilde çıktı alırız


ohe=preprocessing.OneHotEncoder() #üç kolonumuz vardı bunları tr 1 usa2 vs yazmak yarine tek satırda hangisi ise ona 1 yazacak diyerlerine 0 yazacak 
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)
 # tr usa fr
# #[0. 1. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]  bu şekilde çıktı alırız numpy array olarak çıktı alırız

    
 
