# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 09:56:12 2020

@author: ACER
"""

#1.ADIM -- Kütüphaneleri Yükle

import numpy as np #matematiksel araçları içeriyo

import matplotlib.pyplot as plt #grafik oluşturmamıza yardım ediyo

import pandas as pd #veri setini yüklerken, yönetmek istediğimizde kullanırız

#2.ADIM -- Veri Setini Yükle

dataset = pd.read_csv('Data.csv') #veri setini yükledik
#read_csv diyorum çünkü csv dosyasından veri okumaya çalışıyorum

#girdileri X olarak ayarladık, çıktıları y olarak ayarladık
#girdiler --> ülke,yaş, maaş #çıktı satıldı bilgisi

X = dataset.iloc[:,:-1].values #X değişkenine veri setinin içerisindeki ilk 3 kolonu atadık 
#köşeli parantezin içindeki ilk alan satırları gösteriyo,sağında ve solunda değer belirtmeyerek tüm satırları alacağını söylüyorum
#köşeli parantezin virgülden sonraki kısmında hangi kolonları kullanmak istediğimi belirtiyorum. -1 son kolonu gösterir, son kolondan saymaya başlar
#son kolon hariç tüm kolonları(hepsini) al
#virgülden sonra --> son kolon hariç hepsini al
print(X)
#X --> 10 satır 3 kolondan oluşur


y = dataset.iloc[:,-1].values
#tüm satırları al, sadece son kolonu al
print (y)
#y --> 10 satır , tek sütundan oluşur

#3.ADIM -- Eksik Verilerin Kontrolü
#veri setinde girilmemiş bir alan varsa bunlarla ilgili yapabileceğim işlemler var
#1. yöntemde içerisinde boşluk olan satırı veri setinden çıkarabilirim. Ama bu yöntemde veri kaybı oluyo. Ne kadar iyi veri o kadar iyi performans sonuçları demek, elimizden geldiğince verileri kullanmaya çalışmalıyız
#2. yöntemde burdaki boş değerlerin yerine bazı formüller kullanarak doldurmak
#2. yöntemde örneğin ben ortalama alarak yazabilirim, en sık değeri alabilirim. Böyle eksik değeri tamamlayabiliriz
#2. yöntemi kullanıyoruz
#preprocessing = önişlem demek

from sklearn.impute import SimpleImputer
#içerisinden bazı fonksiyonları kullanmak istiyosam bazı özelliklerini kullanmak istiyosam
#Burda sklearn içerisinden sadece Imputer ı aldık

imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #Imputer sınıfından obje oluşturduk
# missing_values = 'NAN' --> boş değerleri veri setinde ne olarak gösterdik onu yaz biz NAN şeklinde gösterdik 
# strategy='mean' --> strateji istiyo ortalama değer, default olarak ortalama kullan
# mean diyerek ortalama almasını istedik
# axis = 0 --> axis ben bu ortalamayı neye göre yapıcam demek , 0 versem kolonlara(sütunlara) göre yapacak, 1 dersem satırların ortalamasını alır. Burda 0 aldık. Sütunlarda ortalama değeri hesaplayıp boş yerlere yazacak

#herhangi bi şeyle ilgili fazla bilgi almak için ctrl+i

imputer=imputer.fit(X[:,1:3]) #oluşturduğum objeyi kullancam
# X için boşluk doldurma yapıyoruz. 1. kolondan 3 e kadar gitsin. Çünkü 0 da boş değer yok ve sayısal değer değil.
#1-2 nolu kolonları al X ten. 3 dahil değil. Yaş ve maaş değerlerini aldık

X[:,1:3]=imputer.transform(X[:,1:3])
#X in tüm satırlarını al. transformunu kullan. 1 den 3 e kadar al
# burda transform boş alan varsa onu veri setimizde ilgili kolonun ortalaması yani mean i ile değiştirecek
print(X)


#4.ADIM -- Kategorik Verilerin Düzenlenmesi
#text i sayılara encode et
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
#sadece 0 indeksli konuma uyguladık -- ülkeye
#hangi veriyi fit ediyosa X'in 0 indeks numarasına sahip olan 1. kolonunu yani kategorik olan kolonunu sayısal verilere çevirecek 
#burda ülkelerin adları yok bunun yerine sayısal veriler var
print(X) 


#Bu şekilde gösterimde şöyle bir program olabilir.
#Tek bir kolonda Almanya, İspanya, Fransa ülkelerine ait bilgileri tutmak yerine her bir tane sınıf için yani Almanya, İspanya, Fransa için ayrı kolon açıp eğer bu değişken Fransa ise 1 değilse 0 yazarak kolon yapısı oluşturuyoruz
#Bu yapı tek bir kolondaki sıralama hatasının önüne geçmemize yardımcı oluyo

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])],remainder='passthrough')
#categorical_features=[0] --> hangi değerler kategorik bunu belirtmem gerekiyo

X = np.array(transformer.fit_transform(X), dtype=np.float)
#X i encoding ediyoruz.
#fit_transform(X) --> hangi değişkene uygulayacağımıza karar veriyoruz
print(X)
#Hangi kolonda 1 varsa ona ait
#0.kolon Fransa, 1.kolon Almanya, 2.kolon İspanya --> hangisi 1 ise o ülkeye ait
#3.kolon yaş, 4.kolon maaş oldu


#çıktıları nümerik olarak ifade ediyoruz 0 ve 1 lerle
#çıktı yes ve no olarak tutuluyo bunları sayısal verilere çevirmemiz gerekiyor. Üzerinde işlem yapabilmek için
labelEncoder_y = LabelEncoder()
y=labelEncoder_y.fit_transform(y)
print(y) #y 0 ve 1 lerle kodlanmış oldu

#VERİLER İŞLEME HAZIR HEPSİ NÜMERİK OLDU

#5.ADIM -- Veri Setini Egitim ve Test olarak 2'ye Böl

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.2, random_state=0)
# train_test_split(X,y --> hangi değişken üzerinde işlem yapıyosan onu yaz --> göndereceğimiz değişkenler
# test_size=0.2 --> %20 sini test, %80 ini eğitim olarak ayarladık --> bu bir hiperparametre bunu kendimiz ayarlıyoruz
# X_train, X_test, y_train, y_test  --> bunlar bize geri dönüyo, değişken olarak tanımladık
# elimizde 10 veri vardı 8 ini traine, 2 sini teste attı


#6.ADIM -- Özellik Ölçeklendirme
#Veriler farklı alanlarla ilgili bilgiler içeriyor.
#Hepsini belli aralığa çekiyoruz

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
print(X_train) #train için - her biri için standartize edilmiş değerler

X_test = sc_X.transform(X_test)
print(X_test) # test için - her biri için standartize edilmiş değerler


#VERİ SETİ ÜZERİNDE ÇALIŞMAYA HAZIR HALE GELİYOR


