# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 09:27:28 2021

@author: ACER
  
"""
    
""" SEVDENUR YILMAZ - 180508036 """

#1 - VERİ ÖNİŞLEME

#1.1.ADIM -- Kütüphaneleri Yükleme

import numpy as np #matematiksel araçları kuallanmak için
import pandas as pd #veri setini yüklerken kullanmak için


#1.2.ADIM -- Veri Setini Yükleme
dataset = pd.read_csv('veriseti.csv') #veri setini oku
#dataset isimli değişken oluşturup, buna veriseti.csv dosyasının içindeki veriler aktarıldı
#400 satırlı, 13 kolonlu alan oluşmuş oldu

X = dataset.iloc[:, :-1].values #girdiler
#print(X)

y = dataset.iloc[:,12].values #çıktı (son kolondaki veri)
#♥print(y)

#1.3.ADIM -- Eksik Verilerin Kontrolünü Yapma
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
"""
!NOT:Yeni sürümlerde Imputer kütüphanesi kullanılmıyor uyarısı aldım, araştırdığımda
bu kullanım şeklinde olduğunu bulup bu kodu kullandım.
"""

imputer = imputer.fit(X[:,0:11]) #oluşturduğum objeyi kullanıyorum

X[:,0:11]=imputer.transform(X[:,0:11])
#print(X) 

#Bu adımda boş değerler ortalama değerler ile dolduruldu

#1.4.ADIM -- Kategorik Verileri Sayısal Verilere Çevirme
from sklearn.preprocessing import LabelEncoder

#Çıktı hasta ve hasta değil olarak tutuluyor, üzerinde işlem yapabilmek için bunları sayısal verilere çevirmemiz gerekiyor. 
labelEncoder_y = LabelEncoder()
y=labelEncoder_y.fit_transform(y)
#print(y) 
#Çıktı 0 ve 1 lerle kodlanmış oldu --> 0 lar hasta , 1 ler hasta değil

#VERİLER İŞLEME HAZIR HEPSİ NÜMERİK OLDU

#1.5.ADIM -- Veri Setini Eğitim ve Test Olarak 2'ye Bölme
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.3, random_state=0)
# %30 unu test, %70 ini eğitim olarak ayarladım
# elimizde 400 veri vardı 280 sini traine, 120 inini teste attı


#1.6.ADIM -- Özellik Ölçeklendirme
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#VERİ ÖNİŞLEME  ile YSA da kullanabilir hale getirdik
#12 kolon-280 satırdan oluşan X_train, 12 kolon-120 satırdan oluşan X_test var
#Tek kolon-120 satırdan oluşan y_test, tek kolon-280 satırdan oluşan y_train var 


#2 - YAPAY SİNİR AĞI
#2.1.ADIM -- Kütüphaneler ve Paketleri Yükleme
import keras
from keras.models import Sequential
from keras.layers import Dense

#2.2.ADIM -- Yapay Sinir Ağını Başlat
classifier = Sequential() #classifier (sınıflandırıcı)

#2.3.ADIM -- Girdi katmanı ve 1. Gizli Katmanı Oluştur
classifier.add(Dense(units=6, input_dim=12,activation='relu', kernel_initializer="uniform")) 

#2.4.ADIM -- 2. Gizli Katman
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform' ))
#input_dim gerek yok keras bunu bizim için kendisi hesaplıyor


#2.5.ADIM -- Çıktı katmanı 
classifier.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform' ))
#Çıktı katmanında 1 değerimiz var, binary sınıflandırma gerçekleştirildi

#2.6.ADIM -- Yapay Sinir Ağı Çalıştır/Derle
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# 2.7.ADIM -- Yapay Sinir Ağını Eğitim Setinde Dene / Eğitmeye Başla
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100) #yapay sinir ağını eğit
#Her 10 veride bir ağırlıklar güncellenecek
#Veri seti üzerinde 100 tur dönecek

#3 - TEST AŞAMASI
""" 
Elimizde test verileri vardı. Bunlardan tahminlemeler gerçekleştireceğiz.
Eğitmiş olduğumuz değerleri bu yapay sinir ağına verip ve yapay sinir ağının 
oluşturmuş olduğu tahmin değerlerin doğru olup olmadığını belirleyeceğiz.
"""
#3.1. ADIM -- Test Verileri için Tahminleri Hesaplama
y_pred=classifier.predict(X_test) #tahmin değerlerini saklayan değişken --> y_pred
y_pred=(y_pred>0.5)  # tahmin değeri (y_pred)y_pred>0.5 ise true(doğru) altında ise false olarak ifade et
#print(y_pred)

#Confusion Matrix
#confisuon matriX kullanarak accuracy hesaplama
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred) 

"""
Etiketlerle bizim yapay sinir ağımızın oluşturmuş olduğu sonuçları karşılaştıracak
Sonuç yanlış olması gerekirken bizim ürettiğimiz yapay sinir ağı sonuçları nasıl?
Bu ikisini confusion matrix bizim için karşılaştıracak.
"""
print(cm)

print("*****")

sonuclar=classifier.evaluate(X_test,y_test)
print("Accuracy: %.2f%%" %(sonuclar[1]*100))

print("*****")

sonuclar2=classifier.evaluate(X_train,y_train)
print("Accuracy: %.2f%%" %(sonuclar2[1]*100))

