# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 08:40:07 2021

@author: ACER
"""

#SINIFLANDIRMA PROBLEMLERİ 
#Bankada kalıp kalmayacağını bulacağız, müşteriyi 2 sınıftan hangisinde olduğunu bulma
#hala müşteri kalanlar 0, bankadan ayrılanlar 1

#1 - VERİ ÖNİŞLEME --> ile YSA da kullanabilir hale getirdik

#1.1.ADIM -- Kütüphaneleri Yükle

import numpy as np #matematiksel araçları içeriyo

import matplotlib.pyplot as plt #grafik oluşturmamıza yardım ediyo

import pandas as pd #veri setini yüklerken


#exited değeri --> y
#Örnek 2 li sınıflandırma örneği 

#1.2.ADIM -- Veri Setini Yükle
dataset = pd.read_csv('veriseti4.csv') #veri setini oku

X=dataset.iloc[:, 3:-1].values
#Müşteri id, soyad, kolon numarası gibi verileri kullanmıyoruz. Bunlar bankada kalıp kalmayacağı ile bilgi vermez
#Sadece ihtiyacımız olanları kullanıyoruz --> ilk 3 kolon --> 0,1,2. indeks
#tüm satırlar gelsin, tüm kolonları almıyoruz 0, 1,2. indeksleri kullanmıyoruz
#3 numaralı indeksten başla
#♥son kolon hariç diğerlerini al
print(X)

y=dataset.iloc[:, 13].values #tüm satırları al, ve son kolonu al (son kolon = exited)
print(y) #1 olan bankadan ayrılanlar, 0 olanlar bankada kalanları gösteriyo


#1.3.ADIM -- Kategorik Verileri Sayısal Verilere Çevirme
#Encoding
#ülkeler cinsiyet vs kategorik alanlar. YSA da kategorik veriler yok. Bunları sayısal veriye çevirmemiz gerekiyo

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#1 ve 2 numaralı indeksler kategorik veri -- bunların 2 sini de sayısal veriye çevir

labelencoder_X_1 = LabelEncoder()

X[:,1] = labelencoder_X_1.fit_transform(X[:,1])#tüm satırlar ve 1.kolonu al
#hangi veriyi transform ediyosak onu parantez içine yazıyoruz
#1.kolonu sayısal yaptık
print(X)

labelencoder_X_2 = LabelEncoder() #cinsiyet için, 0 lar kadınları , 1 ler erkekleri
X[:,2]= labelencoder_X_2.fit_transform(X[:,2])#tüm satırlar ve 2 numaralı kolonu(cinsiyet) al


#kategorik datalar 0 - 1 arasında sayısal büyüklük yok. Ülkelerde 1 2 den büyük vs diyemiyoruz .O yüzden burdaki alanları dummy variable oluşturarak 1 kolonda göstermek yerine 1 den fazla kolonda gösteriyoruz
#12. ders , dk 15
#variable oluşturmak için onehotencoder dan yararlanabiliriz
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1])],remainder='passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.float)

X=X[:,1:] #1 tanesini yani 0 indeksli kolonu çıkardık
print(X) # X veri seti 11 kolondan oluşan bir şekilde kullanıma hazır bulunuyo


#1.4.ADIM -- Veri Setini Eğitim ve Test Olarak Böl

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) 
# Veri setinde 10000 satır var. Bunların 8000 tanesini eğitime, 2000 tanesini teste ayırdık


#1.5.ADIM -- Özellik Ölçeklendirme --> Her bir kolonun skalası farklılıklarının önüne geçmek, bazı verilerin veri setinde etkisinin daha fazla olmasının önüne geçmek, küçük verilerle çalışmak hız açısından iyi 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
print(X_train)

#Özellik ölçeklendirme kullanarak veri setindeki değerlikleri belli aralığa çektik
#elimizdeki tüm veriler belli aralığa çekildi

#VERİ ÖNİŞLEME --> ile YSA da kullanabilir hale getirdik
#Elimde 11 kolondan ve 8000 satırlı oluşan X_train , 11 kolondan ve 2000 satırdan oluşan X_test var
#◘Elimde tek kolondan oluşan 2000 satırlı y_test, tek kolondan oluşan 8000 satırlı y_train var 


#2 - YAPAY SİNİR AĞI
#2.1.ADIM -- kütüphaneler ve paketler
import keras
from keras.models import Sequential
from keras.layers import Dense

#2.2.ADIM -- YSA Başlat
classifier = Sequential() #classifier (sınıflandırıcı)

#2.3.ADIM -- Girdi katmanı ve İlk Gizli Katmanı Oluştur
classifier.add(Dense(units=6, input_dim=11,activation='relu', kernel_initializer="uniform")) #1.gizli katman

#units i değiştirerek performansta artış olup olmadığını bakılabilir
#units --> pozitif integer --> ilk gizli katmandaki her bir node sayısı
#input_dim --> kaç birimden oluştuğu --> örnekte 11 kolonumuz vardı
#activation --> hangi aktivasyon fonksiyonunu kullanacağımızı belirliyoruz --> gizli katmanda relu kullanıyoruz
#kernel_initializer --> ağırlığa başlangıç değeri olarak ne vereceğin 


#2.4.ADIM -- İkinci Gizli Katman
classifier.add(Dense(activation='relu', units=6, kernel_initializer='uniform' ))
#6 node, relu, uniform

#2.5.ADIM -- Çıktı katmanı
classifier.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform' ))
#♣çıktı katmanında 1 değerimiz var, binary sınıflandırma gerçekleştirebilirim

#Katman sayısını 1 artırıp denedik, bu performansta artışa neden olmadı

#2.6.ADIM -- Yapay Sinir Ağı Çalıştır/Derle
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#loss = 'binary_crossentropy' --> 2 kategori olduğu için


# 2.7.ADIM -- Yapay Sinir Ağını Eğitim Setinde Dene / Eğitmeye Başla
classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)
#batch_size = 10 --> her 10 veride bir ağırlıklarımızı güncelliycez
#epoch = 100 --> veri setidnde kaç kez bu işlemi gerçekleştireceğimiz


#3 - TEST AŞAMASI
""" 
Elimizde test verileri vardı. Bunlardan tahminlemeler gerçekleştireceğiz.
Eğitmiş olduğumuz değerleri bu yapay sinir ağına verip ve yapay sinir ağının 
oluşturmuş olduğu tahmin değerlerin doğru olup olmadığını belirleyeceğiz
"""
#3.1. ADIM -- Test Verileri için Tahminleri Hesaplama
y_pred=classifier.predict(X_test) #tahmin değerlerini saklayan değişken --> y_pred
y_pred=(y_pred>0.5)  # tahmin değeri (y_pred)y_pred>0.5 ise true(doğru) altında ise false olarak ifade et

print(y_pred)
#değerler true,false olarak saklanır

#Confusion Matrix
#confisuon matriks kullanarak accuracy hesaplama
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred) 
#etiketle yapay sinir ağının oluşturmuş olduğu sonucu karşılaştıracak


"""
etiketle yapay sinir ağının oluşturmuş olduğu sonucu karşılaştıracak
etiketlerle bizim yapay sinir ağımızın oluşturmuş olduğu sonuçları karşılaştıracak
gerçekte doğruyken ne bulunmuş 
sonuç yanlış olması gerekirken bizim ürettiğimiz yapay sinir ağı sonuçları nasıl
bu ikisini conf matrix bizim için karşılaştıracak
"""
print(cm)
""" 
gerçekte sonucum 0 ken kaç tanesi 0 bulunmuş
gerçekte sonucum 0 ken kaç tanesi 1 bulunmuş
gerçekte sonucum 1 ken kaç tanesi 0 bulunmuş
gerçekte sonucum 1 ken kaç tanesi 1 bulunmuş
bana bunları veriyo
"""








