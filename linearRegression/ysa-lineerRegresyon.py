# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 13:52:09 2020

@author: ACER
"""
#Kişilerin kaç yıl çalıştıklarına bakarak maaşlarını tahmin etme -- Lineer Regresyon

#1.ADIM -- Kütüphaneleri Yükle

import numpy as np #matematiksel araçları içeriyo -- matematiksel işlemler (as kısaltma için)

import matplotlib.pyplot as plt #grafik oluşturmamıza yardım ediyo

import pandas as pd #veri setini yüklerken


#2.ADIM -- Veri Setini Yükle

dataset = pd.read_csv('Salary_Data.csv')
#indeksler 0 dan başlar 
#bi kolon x i diğer kolon y yi -- x=deneyim, y=maaş -- x ten y yi tahminlemeye çalışcaz

X = dataset.iloc[:, :-1].values
#tüm satırları al <-- [:,] 
#sadece son kolon hariç diğer kolonları bana getir, 
#.values --> değerleri al X e ata
print(X)

y = dataset.iloc[:,1].values
# [:,1] --> tüm satırları al, sadece son kolonu al --> maaşları getirdi
#y bağımlı değişken(maaş), x bağımsız değişken(deneyim)
print(y)


#3.ADIM -- Eğitim ve Test Olarak Veri Setini Böl

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3, random_state=0)
# train--eğitim
# test_size=1/3 --> veri setini nasıl böleceğimi belirliyorum
#1/3 ü teste, 2/3 ü eğitime ayırdık --> 30 değeri eğitim ve test için Random olarak böldü

#4.ADIM -- Lineer Regresyon Uygula

from sklearn.linear_model import LinearRegression #Lineer Regresyon

regressor = LinearRegression() #obje oluşturduk
regressor.fit(X_train, y_train) #regresyon eğrisi oluşturuyoruz

#☻Lineer Regresyonda özellik ölçeklendirmeyi kendisi yapıyo

#5.ADIM -- Test Veri Setinden Tahminlemede Bulun

y_pred = regressor.predict(X_test) 
#y_pred --> y tahmin
#X_train lerle eğittik artık X_test ler göndererek tahmin üret ve gerçek değerlerle tahminleri karşılaştır

print(y_pred) #tahminler
print(y_test)


#6.ADIM -- Grafik Çizimi

plt.scatter(X_train, y_train, color='red')
#X_train e karşılık gelen y_train verilerini gösteriyo, kırmızı noktalarla
plt.plot(X_train, regressor.predict(X_train), color='blue') #lineer regresyon eğrisi, mavi çizgi ile
#mavi --> tahminleme, kırmızı --> gerçek değerler(veri setinde olan değerler)
plt.title("Maas vs Deneyim (Egitim veri seti)") #başlık tanımı
plt.xlabel("Deneyim") #x ekseni ismi
plt.ylabel("Maas") #y ekseni ismi
plt.show()#göster

plt.scatter(X_test, y_test, color='red')
#X_test e karşılık gelen y_test verilerini gösteriyo, kırmızı noktalarla
plt.plot(X_test, regressor.predict(X_test), color='blue')
#mavi --> tahminleme, kırmızı --> gerçek değerler(veri setinde olan değerler)
plt.title("Maas vs Deneyim (test veri seti)") #başlık tanımı
plt.xlabel("Deneyim") #x ekseni ismi
plt.ylabel("Maas")#y ekseni ismi
plt.show()#göster