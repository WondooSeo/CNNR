import os
import csv
import sys
import tensorflow as tf
import numpy as np
from PIL import Image


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


if __name__ == '__main__':
    ## Stacking a test dataset ##
    img_path_dir = "Your test image path"
    img_file_list = Load_File_Name(img_path_dir)
    img_data_num = len(img_file_list)

    CP_path_dir = "Your test regression path"
    CP_file_list = Load_File_Name(CP_path_dir)
    CP_data_num = len(CP_file_list)

    if img_data_num == CP_data_num:
        test_img_stacking, test_CP_stacking = Stackin_Data(img_file_list, CP_file_list)

    else:
        sys.stderr.write("Data numbers are not equal! Try again ...")
        exit(1)

    test_img_stacking = np.expand_dims(test_img_stacking, -1)
    test_CP_stacking = np.expand_dims(test_CP_stacking, 1)

    CNNR_path = 'CNNR.h5'
    if (os.path.exists(CNNR_path)):
        CNNR = tf.keras.models.load_model(CNNR_path, compile=True)
        # CNNR.summary()
        print("CNNR exist & loaded ...")

    # test_scores = CNNR.evaluate(test_img_stacking, test_CP_stacking, verbose=1)
    # print("Test Loss : ", test_scores[0])

    pred_CP = CNNR.predict(test_img_stacking)
    pred_CP = np.squeeze(pred_CP)
    test_CP_stacking = np.squeeze(test_CP_stacking)
    diff = list()

    for pred, CP in zip(pred_CP, test_CP_stacking):
        diff.append(abs(pred - CP))

    for CP, pred, now_diff, now_file in zip(test_CP_stacking, pred_CP, diff, img_file_list):
        print(now_file + ' : ' + str(CP) + ' / ' + str(pred) + ' / ' + str(now_diff))

    print(np.mean(diff))
    print(np.std(diff))
