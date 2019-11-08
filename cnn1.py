from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=800,
                         epochs=200,
                         validation_data=test_set,
                         validation_steps=20)

classifier.save('dogcat_bak.h5')

from keras.models import load_model
classifier = load_model('dogcat_bak.h5')

# Making New Predictions
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = test_image/255
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image, steps=1)
if result[:, :] > 0.5:
    value = 'Dog :%1.2f' % (result[0, 0])
    plt.text(20, 62, value, color='red', fontsize=18, bbox=dict(facecolor='white', alpha=0.8))
else:
    value = 'Cat :%1.2f' % (1.0-result[0, 0])
    plt.text(20, 62, value, color='red', fontsize=18, bbox=dict(facecolor='white', alpha=0.8))

plt.imshow(test_image)
plt.show()

# Model Accuracy
x1 = classifier.evaluate_generator(training_set)
x2 = classifier.evaluate_generator(test_set)
print('Training Accuracy  : %1.2f%%     Training loss  : %1.6f' % (x1[1]*100, x1[0]))
print('Validation Accuracy: %1.2f%%     Validation loss: %1.6f' % (x2[1]*100, x2[0]))