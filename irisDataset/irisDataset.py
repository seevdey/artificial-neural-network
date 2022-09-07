# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 09:12:35 2021

@author: ACER
"""
#(encoding ile sayısal veriye çeviririz)

#Veri Setinde iris çiçeğinin türlerini bize gösteriyor
#4 tane özellik var bunlarla çiçeğin türünü belirliyoruz

#1.ADIM --Kütüphaneleri yukle
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np
import pandas as pd

#•veri setinde önişlem yapılmış

# 2.ADIM -- Eğitim Veri Setini Yükle 
egitim_veriseti= pd.read_csv('iris_training.csv')
#120 satır 5 kolon

#Veri Setinin Boş Olup Olmadığının Kontrolü
#egitim_veriseti.isnull() --> #tüm kolonları ve satırları döndürür --> satır satır boş değer olup olmadığını görebiliyorum, boş yoksa false, varsa true yazar
#çok fazla satır olduğu zaman bunu takip etmek zor. 
#egitim_veriseti.isnull().values.any() --> tüm değerlerin içinde boş olup olmadığını gösterir
#egitim_veriseti.isnull().sum() --> hangi alanda boşluk tespit eder


#tüm satırları al, 0. kolondan başla 4. kolona kadar git
egitim_x=egitim_veriseti.iloc[:,0:4].values

#tüm satırları al, 4 nolu kolonu al
egitim_y=egitim_veriseti.iloc[:,4].values

print(egitim_x) #input girdiler
print(egitim_y)

#Encodig uygula -- egitim_y verisine
encoding_egitim_y=np_utils.to_categorical(egitim_y)
print(encoding_egitim_y) #her bir tür için bi kolon oluşturdu

#3.ADIM -- Yapay Sinir Ağı oluştur
model=Sequential()

#Giriş Katmanı ve 1. Gizli katmanı ekle
model.add(Dense(units=4,input_dim=4,activation='relu'))
#input_dim=4 --> girdiler 4 kolondan oluşuyo
#activation='relu' --> hangi aktivasyon fonksiyonunu kullanacağımız

#2. Gizli Katman
model.add(Dense(units=4, activation='relu'))
#input_dim gerek yok keras bunu bizim için kendisi oluşturuyo

#Çıktı Katmanı
model.add(Dense(units=3,activation='softmax'))
#♥kaç sınıf varsa o kadar nöron burda 3


#4.ADIM -- Modeli Çalıştır
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(egitim_x,encoding_egitim_y,epochs=100, batch_size=10) #yapay sinir ağını eğit
#epochs hiperapametre -- değiştirilebilir

#5.ADIM -- Yapay Sinir Ağını Test Et

test_veriseti = pd.read_csv('iris_test.csv') #30 satır 5 kolondan oluşan veri setini aldık

test_x=test_veriseti.iloc[:,0:4].values #tüm satırları al, 0. indeksten başla son kolona kadar git değerleri oku
test_y=test_veriseti.iloc[:,4].values #sadece 4 numaralı kolonu al --> tek kolon 30 satırdan oluşan test_y oluşturuldu


#encoding işlemi test_y için de yapılır
encoding_test_y=np_utils.to_categorical(test_y)
print(test_y) # 0 ve 1 lerden oluşmuş onehotencoding uygulanmış şekli

sonuclar=model.evaluate(test_x,encoding_test_y) #encoding_test_y değilde test_y gönderirsem hata alırım
#evaluate ile değerlendir

print("Accuracy: %.2f%%" %(sonuclar[1]*100))







