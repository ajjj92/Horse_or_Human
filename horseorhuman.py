import os
import zipfile
import urllib.request
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from keras.preprocessing import image


def wget_pics():
    if (not os.path.exists('/home/atte/LUT/AI/horseorhuman/horses')) or (
            not os.path.exists('/home/atte/LUT/AI/horseorhuman/humans')):
        urllib.request.urlretrieve("https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip",
                                   filename="/tmp/horse-or-human.zip")

        zip_ref = zipfile.ZipFile("/tmp/horse-or-human.zip", 'r')
        zip_ref.extractall("/home/atte/LUT/AI/horseorhuman")
        zip_ref.close()


def sort_data():
    # Directory with our training horse pictures
    train_horse_dir = os.path.join('/home/atte/LUT/AI/horseorhuman/horses')
    # Directory with our training human pictures
    train_human_dir = os.path.join('/home/atte/LUT/AI/horseorhuman/humans')
    train_horse_names = os.listdir(train_horse_dir)
    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1 / 255)

    # Flow training images in batches of 128 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        '/home/atte/LUT/AI/horseorhuman/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized 
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    return train_generator


def build_model():
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 300x300 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fifth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['acc'])
    return model


def train_model(model, train_generator):

        DESIRED_ACCURACY = 0.95

        class myCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if (logs.get('acc') > DESIRED_ACCURACY):
                    print('\nReached 95% accuracy, training stopped!')

                    self.model.stop_training = True



        callback1 = myCallback()
        checkpoint_path = "training_1/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        callback2 = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=8,
            epochs=15,
            verbose=1,
            callbacks=[callback1,callback2])

        return model


def main():
    wget_pics()
    checkpoint_path = "training_1/cp.ckpt"
    model = build_model()
    model.load_weights(checkpoint_path)
    running = True

    while running:
        print('*** WELCOME TO AJ NEURALNETWORK ***')
        print('1) Train network with updated data')
        print('2) Use network to predict the output')
        print('3) Graph the network')
        print('4) Exit')

        userinput = int(input(': '))

        if userinput==1:
            train_generator = sort_data()
            model = train_model(model, train_generator)
        elif userinput==2:
            picpath = input('Give the path to the data: ')
            img = image.load_img(picpath, target_size=(300, 300))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            classes = model.predict(images, batch_size=10)
            print(classes[0])
            if classes[0] > 0.5:
                print(picpath + " is a human\n")
            else:
                print(picpath + " is a horse\n")

        elif userinput==4:
            model.save_weights(checkpoint_path)
            print('Program shutting down...')
            break

        else:
            print('Wrong selection')

if __name__ == '__main__':
    main()

