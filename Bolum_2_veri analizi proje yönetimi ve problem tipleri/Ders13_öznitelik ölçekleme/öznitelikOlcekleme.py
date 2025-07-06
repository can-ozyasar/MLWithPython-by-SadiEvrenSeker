#ders 13 özintelik ölçekleme
# Bu derste, özintelik ölçekleme işlemini öğreneceğiz. Özintelik ölçekleme, verilerin belirli bir ölçeğe göre dönüştürülmesi işlemidir. Bu işlem, makine öğrenimi modellerinin daha iyi performans göstermesini sağlar.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




veriler=pd.read_csv('testVEgitim.csv')
print(veriler) #  tüm verileri ekrana yazdırırız






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
#print(sonuc)
print("--------------------------------------------------------")
    
sonuc2 = pd.DataFrame(data=yas, index=range(22),columns=['boy','kilo','yas']) # yas kolonunu dataframe'e çevirdik indexler ile 0 dan 22 ye kadar olan indexlerimizi belirledik
#print(sonuc2)
print("--------------------------------------------------------")


# veriler.iloc[:,-1] # cinsiyet kolonunu alırız sondan 1 öcneki kolon demek  bu kullanım
cinsiyet = veriler.iloc[:,-1].values # cinsiyet kolonunu dataframe'e çevirdik indexler ile 0 dan 22 ye kadar olan indexlerimizi belirledik
#print(cinsiyet)
print("--------------------------------------------------------")
sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=['cinsiyet']) # cinsiyet kolonunu dataframe'e çevirdik indexler ile 0 dan 22 ye kadar olan indexlerimizi belirledik
#print(sonuc3)
print("--------------------------------------------------------")



#dataframeleri değiştirdikten sonra bunları birleştireceğiz
#birleştirme işlemi için concat fonksiyonunu kullanacağız

son_veri = pd.concat([sonuc,sonuc2], axis=1) # axis=1 sütunları birleştirir yanyana ekleme gibi axix=0 satırları birleştirir alt alta ekleme gibi

#print(son_veri)
print("--------------------------------------------------------")

cinsiyetli_son_veri = pd.concat([son_veri,sonuc3], axis=1) # cinsiyet kolonunu da ekledik
#print(cinsiyetli_son_veri)



# DERS 13 öznitelik ölçeklendirmesi verilerin biirbirine daha yakın orantlarda olması için ölçeklendirme işlemi yaparız



from sklearn.model_selection import train_test_split  #veriyi 4 e böleriz eğitim ve test için
# train_test_split fonksiyonu ile veriyi eğitim ve test için böleriz

x_train, x_test, y_train, y_test = train_test_split(cinsiyetli_son_veri.iloc[:,:-1].values,
                                                     cinsiyetli_son_veri.iloc[:,-1].values,
                                                     test_size=0.33, random_state=0) 
# test_size=0.33 ile verinin %33'ünü test için ayırırız random_state=0 ile her seferinde aynı sonucu alırız
#
    


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train) # eğitim verilerini ölçeklendirir daha yakın sayılar ile içerik orantısı konurmuş olur . normalizasyon işlemi yaparız
X_test = sc.transform(x_test) # test verilerini ölçeklendirir
print("Eğitim verileri ölçeklendirilmiş hali:")
print(X_train)
print("Test verileri ölçeklendirilmiş hali:")
print(X_test)
