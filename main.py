import cv2
import os
from emotion_classifier import *
from image_processor import *
from  geometric_helper import *
from feature_detector import *

def create_output_path(dataset_path, emotion_number, index, file_name):
    return "{}/{}/{}_{}.png".format(dataset_path, str(emotion_number), file_name[0:len(file_name) - 4], index)


def create_input_image_path(images_path, input_path, file_path):
    return "{}/{}.png".format(images_path, file_path[len(input_path):len(file_path) - 12])


def create_file_path(path, file_name):
    return "{}/{}".format(path, file_name)


# ПЕРЕДЕЛАТЬ!!!
def create_dataset(input_path, images_path, dataset_path):
    for path, dirs, files in os.walk(input_path):
        for file_name in files:
            file_path = create_file_path(path, file_name)
            file = open(file_path, 'r')

            emotion_number = int(float(file.read()))
            image = cv2.imread(create_input_image_path(images_path, input_path, file_path))

            # templates = get_all_roi(image)
            for i, template in enumerate(templates):
                output_path = create_output_path(dataset_path, emotion_number, i, file_name)
                cv2.imwrite(output_path, template[0])
            file.close()



def create_train_set(dataset_path):
    f = []
    for path, dirs, files in os.walk(dataset_path):
        for file_name in files:
            num = int(path[len(dataset_path)])
            if num >7:
                print("ad;lkfsjf;ldhf suka")
            image = cv2.imread(path + "/" +file_name)
            res = [num]
            res += (list(get_features_from_image(image, get_linear_and_eccentricity_features)[0][0]))
            f.append(res)
    np.savetxt(dataset_path + "dataset.csv", f, delimiter=',', fmt='%f')

def main():
    # input_path = "/home/ilya/Загрузки/Emotion/"
    # images_path = "/home/ilya/Загрузки/cohn-kanade-images/"
    dataset_path = "/home/ilya/Projects/PythonProjects/EmotionRecognition/Dataset/"
    create_train_set(dataset_path)


if __name__ == "__main__":
    main()
