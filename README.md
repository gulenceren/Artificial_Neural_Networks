Artificial Neural Network (ANN);
    Verisetini pandas ile okuyup bir array aktarıyoruz. numpy array formatına dönüştürdük verileri shuffle ile karıştırdık. X ve y değerlerine X in sıfırıncı elemanlardan df nin son columsdan  1 eksiğine kadar veriler eklenir.Son  columsda y olarak eklenmiştir.25 test 75 train olarak ayrılıp train_test_split işlemi uygulandı. MLP’ye de hidden layer 100 olarak ayrıldı bu işlemi 3 defa tekrarlayacaktır.

Deep Neural Network (DNN);
    1.	Oluşturduğumuz menü ile h2o ve keras üzerinden yapılan dnn işlemlerini yapıp verisetlerimizi okuyup h2o ile  classifion isimli colums y olarak atayıp hypers params methodunun içinde epoch sayılarını 100,200,300 olarak belirliyoruz. Hidden layer olacak 3 hidden layer ve 2 deneme yapması için gerekli parametreler girildi. H2O nun GridSerach ile beraber deeplearning kullanılarak en iyi modellere sıralandırıldı. Daha sonra best modelin acc bulundu.

    2.	Train ve test verileri okunup bu veriler üzerinden belirlenen özellik sayılarına göre iloc ile parçalama işlemi yapıldı. Categorical yapısı ile çevrimler sağlandı. Acc değeri 0.62 den küçük olduğu sürece döngüye girip daha sonra 6 optimizer kullanılarak işlemler yürütüldü.16 nöronlu tek bir hidden layer kullanılmıştır 2 nöronlu bir çıkış belirlendi. 150 epochs yaparak 10 denemede eniyi sonuç ve modeli kayıt edildi.

Convolutıonal Neural Networks (CNN);
    1.	Uygulama dosyamız ile aynı konumda bulunması gereken ./Images klasörü yol olarak belirlendi. Kategoriler olarak bu yolun içindeki 3 adet resim klasörünün isimleri kategori olarak belirlendi.
    2.	Bu resimler yeniden 70 size olacak şekilde yeniden düzenlenip grayscale ile ayarlar yapıldı.Sınıf sayısı yapısına göre train ve test dizileri oluşturuldu.
    3.	Bu veriler karıştırılıp train üzerinde kategori yapılarına göre X ve y oluşturulup, test üzerinden test_X ve test_y oluşturuldu. Yeniden reshape ile boyutlandırıldı. Pickle ile yeniden kayıt edilip dosyalar geri açıldı.
    4.	Type(tip) dönüşümü yapıldı.3 kategori olduğu için categorial işlemi yapıldı. Model oluşturulurken X.shape[1:] kadar giriş, 64nöronlu 2 adet hidden layer ve 3 adet çıkış olucak şekilde model oluşturuldu.
    5.	3 optimizers kullanılarak en iyi sonucu veren model kayıt edildi.


h20, xlwt, opencv yüklenmeli