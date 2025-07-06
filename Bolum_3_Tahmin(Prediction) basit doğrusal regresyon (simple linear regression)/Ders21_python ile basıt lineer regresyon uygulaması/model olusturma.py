
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('aylaraGoreSatis.csv')


print(veriler) #?  tüm verileri ekrana yazdırırız

satislar= veriler.iloc[:,1:2].values #satislar kolonunu alırız
satislar2=veriler[['Satislar']] #todo bu yöntem ile de satislar kolonunu alabiliriz .

aylar = veriler.iloc[:,0:1].values #bağımlı değişken
aylar2=veriler[['Aylar']] #todo bu yöntem ile de satislar kolonunu alabiliriz .

print(' aylar :\n ',aylar) #? ay kolonunu alırız
print('satislar: \n' ,satislar) #? satış kolonunu alırız



from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(aylar2,satislar2,test_size=0.33, random_state=0)
#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)









#model oluşturma lineer regresyon ile ;
#! bu bölümde modelimizi oluşturacağız ve eğiteceğiz    



from sklearn.linear_model import LinearRegression
lr = LinearRegression()


lr.fit(X_train,Y_train)#modelimizi eğitiyoruz xtrainden y traine tahmine edecek 
#bakacak  a ayda saış şuymuş b ayda satış şuymuş bunlar arasındaki ilişkiye bakacak ve bir model oluşturacak



#daha sonra bu modelimizi test etmek için x_test i kullanacağız
#ve bu modelimiz ile y_test i tahmin edeceğiz   c ayda sence kaç satış olur

tahmin= lr.predict(X_test)  #xtest den y_test i tahmin edeceğiz
# xtrainden y traini öğrendi  x testten de kendi tahminlerini çıkardı  y test de gerçek değerler bunlarla modelin doğruluğunu konuşabiliriz .

#daha öncesinde standalaştırma işlemi yapmalıyız 


# eğer yapmazsak nasıl olur bakalım 









