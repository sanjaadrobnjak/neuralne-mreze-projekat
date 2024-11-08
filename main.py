import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt

from keras import layers
from keras import Sequential
from keras.optimizers.legacy import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.utils import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import accuracy_score

trainPath = './FastFoodClassification/Train'
validPath = './FastFoodClassification/Valid'
testPath = './FastFoodClassification/Test'

img_size=(64,64)
batch_size=64

Xtrain=image_dataset_from_directory(trainPath,
                                    image_size=img_size,
                                    batch_size=batch_size
                                    )

Xval = image_dataset_from_directory(validPath,
                                    image_size=img_size,
                                    batch_size=batch_size
                                    )

Xtest = image_dataset_from_directory(testPath,
                                    image_size=img_size,
                                    batch_size=batch_size
                                    )

#niz sa imenima klasa
classes = Xtrain.class_names
print(classes)

num_classes=10

#odbirci po klasama u trening skupu
trainData={'Baked Potato':1500, 'Burger':1500, 'Crispy Chicken':1500, 'Donut':1429, 'Fries':1500, 'Hot Dog':1351, 'Pizza':1500, 'Sandwich':1499, 'Taco':1500, 'Taquito':1500}
#odbirci po klasama u validacionom skupu
validData={'Baked Potato':300, 'Burger':300, 'Crispy Chicken':300, 'Donut':300, 'Fries':300, 'Hot Dog':300, 'Pizza':300, 'Sandwich':300, 'Taco':300, 'Taquito':300}
#odbirci po klasama u test skupu
testData={'Baked Potato':100, 'Burger':200, 'Crispy Chicken':100, 'Donut':200, 'Fries':100, 'Hot Dog':200, 'Pizza':200, 'Sandwich':198, 'Taco':100, 'Taquito':100}

plt.figure()
plt.bar(trainData.keys(), trainData.values(), label='Train')
values=list(trainData.values())
plt.ylabel('Broj primeraka')
plt.title('Broj primeraka u svakoj klasi za treniranje')
plt.xticks(range(0, num_classes), rotation='vertical')
plt.subplots_adjust(bottom=0.25)
plt.grid(linestyle='--')
plt.legend()

plt.figure()
plt.bar(validData.keys(), validData.values(), label='Valid')
values=list(validData.values())
plt.ylabel('Broj primeraka')
plt.title('Broj primeraka u svakoj klasi za validaciju')
plt.xticks(range(0, num_classes), rotation='vertical')
plt.subplots_adjust(bottom=0.25)
plt.grid(linestyle='--')
plt.legend()

plt.figure()
plt.bar(testData.keys(), testData.values(), label='Test')
values=list(testData.values())
plt.ylabel('Broj primeraka')
plt.title('Broj primeraka u svakoj klasi za testiranje')
plt.xticks(range(0, num_classes), rotation='vertical')
plt.subplots_adjust(bottom=0.25)
plt.grid(linestyle='--')
plt.legend()

taken_images=[False]*num_classes

plt.figure(figsize=(10, 4))
for img, lab in Xtrain.unbatch().shuffle(buffer_size=1000):
    #ovde dobijamo vrednost klase trenutne slike kao integer, odnosno labele pretvaramo u iterabilne vrednosti
    lab_value=lab.numpy()
    if not taken_images[lab_value]:
        plt.subplot(2, 5, lab_value + 1)
        plt.imshow(img.numpy().astype('uint8'))
        plt.title(classes[lab_value])
        plt.axis('off')
        taken_images[lab] = True

    if all(taken_images):
        break

plt.show()

data_augmentation = Sequential(
  [
    layers.RandomFlip("vertical", input_shape=(img_size[0],
                                                 img_size[1], 3)),
    layers.RandomRotation(0.25),
    layers.RandomBrightness(0.25),
    layers.RandomContrast(0.25)
  ]
)

plt.figure()
for img, lab in Xtrain.take(1):
    plt.title(classes[lab[0]])
    for i in range(num_classes):
        aug_img = data_augmentation(img)
        plt.subplot(2, int(num_classes/2), i+1)
        plt.imshow(aug_img[0].numpy().astype('uint8'))
        plt.axis('off')
plt.show()


model=Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(64, 64, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics='accuracy')

history=model.fit(Xtrain,
                  epochs=50,
                  validation_data=Xval,
                  verbose=0)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplot(121)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()
plt.subplot(122)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.show()

labels = np.array([])
pred = np.array([])
for img, lab in Xval:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))

print('Tačnost modela je: ' + str(100*accuracy_score(labels, pred)) + '%')

cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
plt.figure(figsize=(12, 10))
cmDisplay.plot(xticks_rotation='vertical')
plt.subplots_adjust(bottom=0.23)
plt.show()

import random

random_samples = random.sample(list(Xtest.unbatch().as_numpy_iterator()), 10)

sample_images = []
sample_labels = []
predicted_labels = []

for img, lab in random_samples:
    sample_images.append(img)
    sample_labels.append(lab)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    predicted_labels.append(predicted_label)

plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(sample_images[i].astype('uint8'))
    plt.title(f"Actual: {classes[sample_labels[i]]}\nPredicted: {classes[predicted_labels[i]]}")
    plt.axis('off')

plt.show()