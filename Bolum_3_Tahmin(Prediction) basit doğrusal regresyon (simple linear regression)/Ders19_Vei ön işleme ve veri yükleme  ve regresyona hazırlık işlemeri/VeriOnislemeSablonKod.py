
 # bölüm 2 de ders 14  de kullanılan şablon kodlar bu şekilde diğer derslerde de  kullanılacak olna temel
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('aylaraGoreSatis.csv')


print(veriler) #?  tüm verileri ekrana yazdırırız

satislar= veriler.iloc[:,1:2].values #satislar kolonunu alırız
satislar2=veriler[['Satislar']] #todo bu yöntem ile de satislar kolonunu alabiliriz .

#! veriler.iloc[:,1:2].values işleminde : tüm satırları al demek ve , den sonrası :2 ise 1. sütundan 2. sütuna kadar olan değerleri al demektir.
aylar = veriler.iloc[:,0:1].values #bağımlı değişken
aylar2=veriler[['Aylar']] #todo bu yöntem ile de satislar kolonunu alabiliriz .

print(' aylar :\n ',aylar) #? ay kolonunu alırız
print('satislar: \n' ,satislar) #? satış kolonunu alırız


#! veriler.iloc[:,0:1].values işleminde : tüm satırları al demek ve , den sonrası :1 ise 0. sütundan 1. sütuna kadar olan değerleri al demektir.








#?verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(aylar2,satislar2,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)



