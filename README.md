# Baigiamasi-darbas
Vaiva Tv  baigiamasis darbas

Giluminio Mokymosi Modelis su CIFAR-10
Apžvalga
Šiame projekte sukuriamas giluminio mokymosi modelis, naudojantis CIFAR-10 duomenų rinkiniu vaizdų klasifikavimui. Modelis naudoja konvoliucinius neuroninius tinklus (CNN) ir įtraukia kelias pažangias technikas, tokias kaip duomenų išplėtimas (Data Augmentation), hiperparametrų optimizavimas ir mokymosi greičio tvarkaraščio reguliavimas. Projektas apima kelis etapus, tokius kaip modelio mokymas, našumo vertinimas ir klaidų analizė.

Funkcijos ir Patobulinimai
1. Giluminio Mokymosi Modelis
Konvoliuciniai sluoksniai: Išgaunami požymiai iš CIFAR-10 vaizdų naudojant keletą konvoliucinių sluoksnių.
Duomenų išplėtimas: Modelis treniruojamas naudojant duomenų išplėtimo technikas, tokias kaip atsitiktinis apvertimas, sukimas, priartinimas, vertikalus ir horizontalus perkėlimas bei kontrasto keitimas, siekiant pagerinti modelio gebėjimą generalizuoti.
Reguliavimas: Pridėti Dropout sluoksniai po kiekvieno konvoliucinio sluoksnio ir prieš išvestinį sluoksnį, siekiant sumažinti perpratimą.
Vertinimas: Modelis įvertina našumą naudodamas tikslumą (accuracy), nuostolį (loss) ir papildomus rodiklius, tokius kaip accuracy_percent ir loss_rounded.
2. Hiperparametrų Optimzavimas
Keras Tuner: Integruota keras-tuner biblioteka, padedanti automatizuoti hiperparametrų optimizavimą ir gerinti modelio našumą.
3. Mokymosi Greičio Tvarkaraštis
Įgyvendintas eksponentinis mokymosi greičio mažinimas per epochą, kad modelis geriau konverguotų.
4. Confusion Matrix ir Klaidų Analizė
Po mokymo modelis sugeneruoja klaidų matrica (confusion matrix) vizualizacijai.
Parodomos klaidingai klasifikuotos nuotraukos, kad būtų galima atlikti tolesnę analizę.
Įdiegimas
Norėdami įdiegti reikiamas bibliotekas, naudokite šią komandą:

bash
!pip install keras-tuner
Duomenų Rinkinys
Modelis treniruojamas naudojant CIFAR-10 duomenų rinkinį, kuris susideda iš 60,000 32x32 spalvotų vaizdų, padalintų į 10 klasių, kiekvienoje klasėje yra 6,000 vaizdų. Duomenų rinkinys suskirstytas į 50,000 treniravimo vaizdų ir 10,000 testavimo vaizdų.

Kodo Struktūra
1. Duomenų Paruošimas
CIFAR-10 duomenys užkraunami ir normalizuojami prieš pateikiant juos modeliui:

python
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
2. Duomenų Išplėtimas
Duomenų išplėtimas taikomas modeliui, kad būtų padidintas jo atsparumas:

python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.2),
])
3. Modelio Architektūra
Modelis naudoja konvoliucinius sluoksnius, pooling, dropout ir visiškai sujungtus sluoksnius:

python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Lambda(lambda x: data_augmentation(x)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax'),
])
4. Modelio Kompiliavimas
Modelis kompiliuojamas su Adam optimizatoriumi ir „sparse categorical cross-entropy“ nuostoliu:

python
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
5. Modelio Treniravimas
Modelis treniruojamas su mokymosi greičio tvarkaraščiu, kuris mažina mokymosi greitį po kiekvienos epochos:

python
Kopijuoti
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)
history = model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))
6. Modelio Vertinimas
Modelio našumas įvertinamas naudojant testavimo duomenis:

python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')
7. Vizualizacija
Tikslumo ir Nuostolių Kreivės: Grafikai, rodantys treniravimo ir validacijos tikslumą/nuostolį per epochas.
Confusion Matrix: Vizualinis klaidų klasifikavimo atvaizdavimas.
Klaidų Analizė: Parodomos klaidingai klasifikuotos nuotraukos.

python
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
Confusion Matrix ir Klaidų Analizė:

python
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(test_labels, y_pred_classes)
Pavyzdinis Išvedimas
Tikslumo ir Nuostolių Kreivės: Grafikai, rodantys treniravimo ir validacijos tikslumą/nuostolį per epochas.
Confusion Matrix: Klaidų klasifikavimo matrica vizualiai pavaizduota.
Klaidų Nuotraukos: Nuotraukos, kurios buvo klaidingai klasifikuotos modelio.
Išvada
Šis projektas demonstruoja, kaip sukurti tvirtą giluminio mokymosi modelį vaizdų klasifikavimui naudojant CIFAR-10 duomenų rinkinį. Pagrindinės funkcijos, tokios kaip duomenų išplėtimas, dropout, hiperparametrų optimizavimas ir mokymosi greičio tvarkaraštis, padeda pagerinti modelio tikslumą ir atsparumą. Confusion matrix ir klaidų analizė suteikia papildomų įžvalgų apie modelio veikimą.
