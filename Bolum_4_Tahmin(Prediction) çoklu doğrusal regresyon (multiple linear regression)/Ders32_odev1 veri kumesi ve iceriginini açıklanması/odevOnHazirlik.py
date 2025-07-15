
#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('odev_tenis_Veri.csv')
#pd.read_csv("veriler.csv")



#veri on 
# öncelikle windy ve play columns'larını numeric hale getireceğiz.
#ohe ile yaparsak bu durumda dummy variable tuzağına düşeriz.
#Bu yüzden label encoder kullanacağız.
# outlook da da  sunny 1 rainy 2 overcast 3 şeklinde sayısal hale getirmeyeceğiz bu durum da hatalı olur .
#bu yüzden outlook u onehot encoder ile yapacağız.




#encoder:  Kategorik -> Numeric
#      
#                                            **************      1     *****************
from sklearn.preprocessing import LabelEncoder

play=veriler.iloc[:,-1:].values
print("Play: \n",play)
#kolonu label encodig ile numeric hale getireceğiz.


le = LabelEncoder()
play[:,0] = le.fit_transform(play[:,0])
print(play) # no 0  yes 1 oldu




#? şimdi de windy için aynı işlemi yapalım.
from sklearn.preprocessing import LabelEncoder  # Import ekleyin

windy=veriler.iloc[:,-2:-1].values
print("Windy: \n",windy)
#kolonu label encodig ile numeric hale getireceğiz.


le2 = LabelEncoder()
windy = le2.fit_transform(windy)
windy = windy.reshape(-1,1)  # Yeniden şekillendir
print("LE windy \n", windy)  # True->1, False->0 olacak



#                                             ******************        2       ******************* 
#bir trick ile tüm bu encode işlemlerini tek seferde yapabiliriz.
#veriler2 = veriler.apply(LabelEncoder().fit_transform) 


veriler2 = veriler.apply(LabelEncoder().fit_transform)

print("veriler2 \n", veriler2)

#burada tüm tabloyu label encoder ile numeric hale getiriyor yani outlook, windy ve play kolonlarını.
#ancak outlook kolonunu onehot encoder ile yapacağız.
#nümerik olan temreture ile humidity kolonlarını ise label encoder ile numeric hale getirmeyeceğiz.
#aksi halde derece olması gereken sayılara 0 1 2 3 ..... değerlerini verir eşleşenlere bu hatalı olur 



c = veriler2.iloc[:,:1]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print("c:",c)

# şimdi de istediğimiz verilere göre son durumda kullanılacak verilerimizi oluşturalım.
# 
havadurumu = pd.DataFrame(data = c, index = range(14), columns=['o','r','s']) # burada o sunny r rainy s overcast olarak onehot encoder ile oluşturduğumuz kolonları isimlendirdik.
print("havadurumu \n",havadurumu)
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1) #  temreture ve humidity kolonlarını ekledik.
print(sonveriler)
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler], axis = 1) # windy ve play kolonlarını ekledik.
print(sonveriler)






#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)
# son kolona kadar olanlar bağımsız değişken son kolonda bulunan play ise bağımlı değişken.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)

print(y_pred)

import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

sonveriler = sonveriler.iloc[:,1:]

import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)







