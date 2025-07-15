#ders 13 özintelik ölçekleme
# Bu derste, özintelik ölçekleme işlemini öğreneceğiz. Özintelik ölçekleme, verilerin belirli bir ölçeğe göre dönüştürülmesi işlemidir. Bu işlem, makine öğrenimi modellerinin daha iyi performans göstermesini sağlar.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# veri onisleme
#veri yükleme 
# 

veriler=pd.read_csv('veriler.csv')
print(veriler) #  tüm verileri ekrana yazdırırız






from sklearn.impute import SimpleImputer


imputer=SimpleImputer(missing_values=np.nan,strategy ='mean')
 #dosyada nan olan değerleri mean yani ortalamaneyse o olarak yazar günceller 


c=veriler.iloc[:,-1:].values
# print(yas)


ulke =veriler.iloc[:,0:1].values
# print(ulke) #ülke kolonunu alırız







from sklearn import preprocessing

le=preprocessing.LabelEncoder()


ulke[:,0] = le.fit_transform(veriler.iloc[:,0]) # ulke kolonundaki verileri etiket kodlama ile sayısal verilere dönüştürürüz
# fit_transform ile hem fit hem de transform işlemini yaparız 

print(ulke)


ohe=preprocessing.OneHotEncoder() #üç kolonumuz vardı bunları tr 1 usa2 vs yazmak yarine tek satırda hangisi ise ona 1 yazacak diyerlerine 0 yazacak 
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)




c[:,-1] = le.fit_transform(veriler.iloc[:,-1]) # ulke kolonundaki verileri etiket kodlama ile sayısal verilere dönüştürürüz
# fit_transform ile hem fit hem de transform işlemini yaparız 

print(c)


ohe=preprocessing.OneHotEncoder() #üç kolonumuz vardı bunları tr 1 usa2 vs yazmak yarine tek satırda hangisi ise ona 1 yazacak diyerlerine 0 yazacak 
c=ohe.fit_transform(c).toarray()
print(c)


#dataframeleri değiştirdikten sonra bunları birleştireceğiz
#birleştirme işlemi için concat fonksiyonunu kullanacağız






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


print(list(range(222))) #bu sayı aralığında lise oluştururuz
print("--------------------------------------------------------") 