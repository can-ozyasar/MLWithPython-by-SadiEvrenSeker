
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('aylaraGoreSatis.csv')



satislar2=veriler[['Satislar']] #todo bu yöntem ile de satislar kolonunu alabiliriz .

aylar2=veriler[['Aylar']] #todo bu yöntem ile de satislar kolonunu alabiliriz .



from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(aylar2,satislar2,test_size=0.33, random_state=0)
#verilerin olceklenmesi
###
#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#X_train = sc.fit_transform(x_train)
#X_test = sc.fit_transform(x_test)
#Y_train=sc.fit_transform(y_train)
#Y_test=sc.fit_transform(y_test)




#model oluşturma lineer regresyon ile ;
#! bu bölümde modelimizi oluşturacağız ve eğiteceğiz    




from sklearn.linear_model import LinearRegression
lr = LinearRegression()


lr.fit(x_train,y_train)#modelimizi eğitiyoruz xtrainden y traine tahmine edecek 
#bakacak  a ayda saış şuymuş b ayda satış şuymuş bunlar arasındaki ilişkiye bakacak ve bir model oluşturacak



#daha sonra bu modelimizi test etmek için x_test i kullanacağız
#ve bu modelimiz ile y_test i tahmin edeceğiz   c ayda sence kaç satış olur

tahmin= lr.predict(x_test)  #xtest den y_test i tahmin edeceğiz
# xtrainden y traini öğrendi  x testten de kendi tahminlerini çıkardı  y test de gerçek değerler bunlarla modelin doğruluğunu konuşabiliriz .




#! bu bölümde modelimizi GÖRSELLEŞTİRECEĞİZ    


# plt.plot(x_train, y_train, color='red') #bu kullanım hatalı çünkü veriler sıralı değil ayları artan *index * sırayla sıralamamız gerekli 
# şuanda randon state 0 olarak oluşturmuştuk  x_train, x_test,y_train,y_test = train_test_split(aylar2,satislar2,test_size=0.33, random_state=0)



#burada sıralama indexe göre yapılıyor ayların küçük yada büyüğüne göre değil öyle olsa en küçük aya en küçük satış değeri gelirdi
#  bu da hatalı olur du

x_train= x_train.sort_index() #x_train index artan sıraya göre sıraladık
y_train= y_train.sort_index() #y_train index artan sıraya göre sıraladık

plt.plot(x_train,y_train) #x_test ve tahmin i
plt.plot(x_test,lr.predict(x_test)) #  her tahmin değerleri lineer regreston ile yaptığımız

plt.title("aylara göre satışlar",color="red")
plt.xlabel("aylar" ) # x kolonu için isimlendirme 
plt.ylabel("satış") # y kolonu için isimlendirme 




