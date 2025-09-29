import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

def main():
    choice = input("select mode, train, retrain or test ")
    match choice:
        case "retrain":
            retrain_network()
        case "train":
            really = input("are you sure?")
            if really == "beast":
                train_network()
        case "test":
            test_network()
        case _:
            print("???")
    
    

def train_network():
    #change image res to 150x150
    img_size = (150, 150)
    #pass 32 samples into the network at once
    batch_size = 32

    #rescaling pixel values between 0 and 1, and slightly changing images to help model cope better with external images
    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.15, zoom_range = 0.15, horizontal_flip = True)

    #for the validation set, no change needs to happen other than pixel rescaling
    validation_datagen = ImageDataGenerator(rescale = 1./255)

    # creating training and validation sets, class mode is binary because it is either a cat or a dog.
    training_set = train_datagen.flow_from_directory(r'C:\Users\keizu\Downloads\dataset\training_set', target_size = img_size, batch_size = batch_size, class_mode = 'binary')

    validation_set = validation_datagen.flow_from_directory(r'C:\Users\keizu\Downloads\dataset\validation_set', target_size = img_size, batch_size = batch_size, class_mode = 'binary')

    # defining a checkpoint
    checkpoint_path = "first_model_checkpoint.h5"
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',  # basing the best model upon validation training set accuracy as opposed to just accuracy(which is within the data set)
        save_best_only=True,  # save only the best model
        mode='max',  
        verbose=1) # updates on the model being saved at checkpoints
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    # now creating the actual neural network by defining its architecture.
    model = keras.Sequential([
        #input shape is 150x150x3 as that is our image res + rgb. 
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        #multiple convolutional layers used for better feature detection
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        #maxpooling used to spatially down-sample the dataset
        keras.layers.MaxPooling2D((2, 2)),
        #turning data into a 1d array
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        #squishing the values between 0 and 1
        keras.layers.Dense(1, activation='sigmoid')])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])
    model.fit(training_set,epochs=50,validation_data=validation_set, callbacks=[checkpoint_callback, early_stopping])

def retrain_network():
    #change image res to 150x150
    img_size = (150, 150)
    #pass 32 samples into the network at once
    batch_size = 32

    #rescaling pixel values between 0 and 1, and slightly changing images to help model cope better with external images
    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.15, zoom_range = 0.15, horizontal_flip = True)

    #for the validation set, no change needs to happen other than pixel rescaling
    validation_datagen = ImageDataGenerator(rescale = 1./255)

    # creating training and validation sets, class mode is binary because it is either a cat or a dog.
    training_set = train_datagen.flow_from_directory(r'C:\Users\keizu\Downloads\dataset\training_set', target_size = img_size, batch_size = batch_size, class_mode = 'binary')

    validation_set = validation_datagen.flow_from_directory(r'C:\Users\keizu\Downloads\dataset\validation_set', target_size = img_size, batch_size = batch_size, class_mode = 'binary')
    
    checkpoint_path = "first_model_checkpoint.h5"
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    # define a checkpoint
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',  # basing the best model upon validation training set accuracy as opposed to just accuracy(which is within the data set)
        save_best_only=True,  # save only the best model
        mode='max',  
        verbose=1)
    num = int(input("how many epochs? "))
    # load the trained model
    model = tf.keras.models.load_model(checkpoint_path)
    model.fit(training_set,epochs=num,validation_data=validation_set, callbacks=[checkpoint_callback, early_stopping])

def test_network():
    #loading model
    model = tf.keras.models.load_model("first_model_checkpoint.h5")

    # taking image as input
    image = cv2.imread("cat_or_dog_2.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

    #resize/normalise
    image = cv2.resize(image, (150, 150))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    # use the model to make a prediction
    predictions = model.predict(image)

    #get the confidence for the highest probability class
    dog_chance = predictions[0][0]

    # Check if the chance for "cat" is below the threshold
    if dog_chance > 0.5:
        # if chance < 0.5, predict dog
        predicted_name = "dog"
        chance = dog_chance
    else:
        # if the chance is > 0.5, predict cat
        predicted_name = "cat"
        chance = 1 - dog_chance

    print("The model predicts that the image is a:", predicted_name)
    print(f"The model's confidence for {predicted_name}: {chance:.2%}")

if __name__ == "__main__":
    main()