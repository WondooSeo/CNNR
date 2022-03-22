## Setup ##
import os
import csv
import sys
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.losses import Huber
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split


def Load_File_Name(passed_dir):
    file_list = []
    for (root, directories, files) in os.walk(passed_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    return file_list


def Stackin_Data(img_file_list, CP_file_list):
    img_file_num = len(img_file_list)
    # CP_file_num = len(CP_file_list)
    img_stack = []
    CP_stack = []
    count = 0

    for img, CP in zip(img_file_list, CP_file_list):
        # Stack image
        np_img = np.asarray(Image.open(img)) / 255.
        img_stack.append(np_img)

        # Stack collapse percentage csv
        reader = list(csv.reader(open(CP)))
        now_b = np.squeeze(reader)
        now_b = [float(now_b)]
        now_b = np.transpose(np.reshape(np.asarray(now_b), -1))
        CP_stack.append(now_b)
        count += 1

        print(str(count) + " / " + str(img_file_num) + " Image & CP Stack Finished ...")

        # Debugging code
        # if count == 1000:
        #     break

    return img_stack, CP_stack


def CNNR_model():
    CNNR_model = Sequential(name='CNNR_Model')
    # Stride = 1
    CNNR_model.add(Conv2D(filters=10, kernel_size=(4, 4), activation='relu', padding='same', input_shape=(128, 128, 1)))
    CNNR_model.add(BatchNormalization())
    CNNR_model.add(MaxPooling2D(2, 2))
    CNNR_model.add(Conv2D(filters=10, kernel_size=(4, 4), activation='relu', padding='same'))
    CNNR_model.add(BatchNormalization())
    CNNR_model.add(MaxPooling2D(2, 2))
    CNNR_model.add(Conv2D(filters=10, kernel_size=(4, 4), activation='relu', padding='same'))
    CNNR_model.add(BatchNormalization())
    CNNR_model.add(MaxPooling2D(2, 2))
    CNNR_model.add(Conv2D(filters=10, kernel_size=(4, 4), activation='relu', padding='same'))
    CNNR_model.add(BatchNormalization())
    CNNR_model.add(MaxPooling2D(2, 2))
    CNNR_model.add(Flatten())
    CNNR_model.add(Dense(64, activation="relu"))
    CNNR_model.add(Dropout(0.2))
    CNNR_model.add(Dense(32, activation="relu"))
    CNNR_model.add(Dropout(0.2))
    CNNR_model.add(Dense(16, activation="relu"))
    CNNR_model.add(Dropout(0.2))
    CNNR_model.add(Dense(1, activation="sigmoid"))
    CNNR_model.compile(optimizer='adam', loss=Huber(), metrics=['mae', 'mse'])
    CNNR_model.summary()
    return CNNR_model


if __name__ == '__main__':

    image_method = 'GREIT'

    ## Stacking a dataset ##
    img_path_dir = 'Write your dir'
    img_file_list = Load_File_Name(img_path_dir)
    img_data_num = len(img_file_list)

    CP_path_dir = 'Write your dir'
    CP_file_list = Load_File_Name(CP_path_dir)
    CP_data_num = len(CP_file_list)

    if img_data_num == CP_data_num:
        img_stacking, CP_stacking = Stackin_Data(img_file_list, CP_file_list)

    else:
        sys.stderr.write("Data numbers are not equal! Try again ...")
        exit(1)

    # Expand dims to fit the input of encoder
    img_stacking = np.expand_dims(img_stacking, -1)
    CP_stacking = np.expand_dims(CP_stacking, 1)

    x_train, x_test, y_train, y_test = train_test_split(img_stacking, CP_stacking, shuffle=True, test_size=0.15)
    print("Data split Finished ...")

    CNNR = CNNR_model()
    history = CNNR.fit(x_train, y_train, validation_split=0.15, epochs=100, batch_size=100, verbose=1, shuffle=True)

    # Show plot of loss and accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('CNNR Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()

    CNNR_path = 'CNNR.h5'
    CNNR.save(CNNR_path)
