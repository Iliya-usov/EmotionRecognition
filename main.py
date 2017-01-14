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


def main():
    # input_path = "/home/ilya/Загрузки/Emotion/"
    # images_path = "/home/ilya/Загрузки/cohn-kanade-images/"
    # dataset_path = "/home/ilya/Projects/PythonProjects/EmotionRecognition/Dataset/"
    # create_dataset(input_path, images_path, dataset_path)

    image = cv2.imread("/home/ilya/Загрузки/f.png")

    image = histogram_equalization(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = get_roi_of_faces(gray_image)
    for im in result:
        h, w = im[0][2].shape
        gradX = cv2.Canny(im[0][0
                          ], 50, 100)
        cv2.imshow("s", gradX)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def t():
    image = cv2.imread("/home/ilya/Загрузки/f.png")

    image = histogram_equalization(image)
    image = get_gray_image(image)
    res = get_roi_of_faces(image)
    scale = 1
    delta = 0
    ddepth = cv2.CV_64F
    for im in res:
        for i, roi in enumerate(im[0]):
            img = roi
            grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
            grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)

            img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

            img = binarization(img, 70)
            cv2.imshow("roi{}".format(i), img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image = cv2.imread("/home/ilya/Загрузки/f.png")

    res = get_features_from_image(image, get_linear_and_eccentricity_features)
    print(res)
