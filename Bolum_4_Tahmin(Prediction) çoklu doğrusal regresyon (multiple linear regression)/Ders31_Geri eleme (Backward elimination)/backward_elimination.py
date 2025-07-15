
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

#encoder: Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
Yas = veriler.iloc[:,1:4].values



print(ulke)


ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#encoder: Kategorik -> Numeric
c = veriler.iloc[:,-1:].values
print(c)


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(c)


ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)



#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1], index = range(22), columns = ['cinsiyet'])
print(sonuc3)



#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression
regrression = LinearRegression()
regrression.fit(x_train,y_train)    

y_pred=regrression.predict(x_test)
print(y_pred)


#boy kolonunu seçerek tahmin ettirelim 
#boy kolonu seçme 
# s2 yi boy kolonu boy kolonu sağı ve solu olarak 3 e böldük 

boy=s2.iloc[:,3:4].values
print(boy)

sol=s2.iloc[:,:3].values
print(sol)
sag=s2.iloc[:,4:].values
print(sag)

#aradan boy kolonu çıkmış s2 verisi oluşturduk 
# öce ayırdık sonra birleşitrdik
veri=pd.concat([sol,sag], axis=1)

x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.3,random_state=0)

# model oluşturalım

regrression2 = LinearRegression()
regrression2.fit(x_train,y_train)    
y_pred2=regrression2.predict(x_test)
print(y_pred2)



# Backward Elimination
# Bağımlı ve bağımsız değişkenlerin ayrılması

import statsmodels.api as sm

#yapılacak işlem bir dizi oluşturup yüm elemanları eklemek
# daha sonra bu diziden stunları çıkararak en iyi sonucu veren modeli bulmak
#hangi değişken sistemi daha yanlış sonuç ürettiriyor onu bulacağız .



X=np.append(arr=np.ones((22,1)).astype(int), values=veri, axis=1  ) #axis=1 # Bağımsız değişkenler için 1 stunu ekledik

X_l = veri.iloc[:,[0,1,2,3,4,5]].values # Bağımsız değişkenler
X_l = np.array(X_l, dtype=float) # float tipine dönüştürdük
model=sm.OLS(boy,X_l).fit() # OLS: Ordinary Least Squares modelin başarısını ölçmek için kullanılır rapor gibidir.
# p değerleri de bu raporda bulunur  p değeri ne kadar düşük ise o kdr iyi bir modeldir
#en büyük p değeri olan değişkeni çıkaracağız  GERİ CLSELEME YÖNTEMİ OLDUĞU İÇİN
print(model.summary())




X_l = veri.iloc[:,[0,1,2,3,5]].values # # Bağımsız değişkenlerden 4 in de p değeri olduğu için sildik
X_l = np.array(X_l, dtype=float) 
model=sm.OLS(boy,X_l).fit() 
print(model.summary())



X_l = veri.iloc[:,[0,1,2,3]].values # Bağımsız değişkenlerden 5 in de p değeri olduğu için sildik
X_l = np.array(X_l, dtype=float) 
model=sm.OLS(boy,X_l).fit() 
print(model.summary())




