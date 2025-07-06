#ders 12 kümelerinin ayrılması
# Kategorik veriler üzerinde işlem yaparken, verilerin doğru bir şekilde kodlanması ve eksik verilerin uygun bir şekilde doldurulması önemlidir.
# örnek olarak verinin yüzde 70 i eğitim için 30 u test için ayrılması gibi bu sayede daha doğru sonuçlar elde edebiliriz.
# Bu işlem, modelin genel performansını artırabilir ve aşırı öğrenmeyi önleyebilir. buh yüzden verileri bölme ve birleştirme işlemleri çok önemlidir.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




veriler=pd.read_csv('testVEgitim.csv')
print(veriler) #  tüm verileri ekrana yazdırırız




# DERS 12 test ve eğitim kümelerine ayrılması 


from sklearn.impute import SimpleImputer


imputer=SimpleImputer(missing_values=np.nan,strategy ='mean')
 #dosyada nan olan değerleri mean yani ortalamaneyse o olarak yazar günceller 


yas=veriler.iloc[:,1:4].values
# print(yas)


ulke =veriler.iloc[:,0:1].values
# print(ulke) #ülke kolonunu alırız

from sklearn import preprocessing

le=preprocessing.LabelEncoder()


ulke[:,0] = le.fit_transform(veriler.iloc[:,0]) # ulke kolonundaki verileri etiket kodlama ile sayısal verilere dönüştürürüz
# fit_transform ile hem fit hem de transform işlemini yaparız 

# print(ulke)


ohe=preprocessing.OneHotEncoder() #üç kolonumuz vardı bunları tr 1 usa2 vs yazmak yarine tek satırda hangisi ise ona 1 yazacak diyerlerine 0 yazacak 
ulke=ohe.fit_transform(ulke).toarray()
# print(ulke)


sonuc =pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','usa']) # ulke kolonunu one hot encoding ile kodladık ve dataframe'e çevirdik indexler ile 0 dan 22 ye kadar olan indexlerimizi belirledik
print(sonuc)
print("--------------------------------------------------------")

sonuc2 = pd.DataFrame(data=yas, index=range(22),columns=['boy','kilo','yas']) # yas kolonunu dataframe'e çevirdik indexler ile 0 dan 22 ye kadar olan indexlerimizi belirledik
print(sonuc2)
print("--------------------------------------------------------")


# veriler.iloc[:,-1] # cinsiyet kolonunu alırız sondan 1 öcneki kolon demek  bu kullanım
cinsiyet = veriler.iloc[:,-1].values # cinsiyet kolonunu dataframe'e çevirdik indexler ile 0 dan 22 ye kadar olan indexlerimizi belirledik
print(cinsiyet)
print("--------------------------------------------------------")
sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=['cinsiyet']) # cinsiyet kolonunu dataframe'e çevirdik indexler ile 0 dan 22 ye kadar olan indexlerimizi belirledik
print(sonuc3)
print("--------------------------------------------------------")



#dataframeleri değiştirdikten sonra bunları birleştireceğiz
#birleştirme işlemi için concat fonksiyonunu kullanacağız

son_veri = pd.concat([sonuc,sonuc2], axis=1) # axis=1 sütunları birleştirir yanyana ekleme gibi axix=0 satırları birleştirir alt alta ekleme gibi

print(son_veri)
print("--------------------------------------------------------")

cinsiyetli_son_veri = pd.concat([son_veri,sonuc3], axis=1) # cinsiyet kolonunu da ekledik
print(cinsiyetli_son_veri)



# DERS 12 kümelerin ayrılması



from sklearn.model_selection import train_test_split  #veriyi 4 e böleriz eğitim ve test için
# train_test_split fonksiyonu ile veriyi eğitim ve test için böleriz

X_train, X_test, y_train, y_test = train_test_split(cinsiyetli_son_veri.iloc[:,:-1].values,
                                                     cinsiyetli_son_veri.iloc[:,-1].values,
                                                     test_size=0.33, random_state=0) 
# test_size=0.33 ile verinin %33'ünü test için ayırırız random_state=0 ile her seferinde aynı sonucu alırız
#
    


    # veriyi  dikey eksende bağımlı ve bağımsız değişkenler olarak ayırdık 
    #sonra bu veriyi eğitim ve test için böldük  toplam 4 parça elde ettik
print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:\n", y_train)
print("y_test:\n", y_test)

