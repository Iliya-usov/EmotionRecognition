import cv2
import os
from emotion_classifier import *
from image_processor import *
from geometric_helper import *
from feature_detector import *
from sklearn.ensemble import RandomForestClassifier
from numpy import savetxt, loadtxt
import pickle


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
            if num > 7:
                print("ad;lkfsjf;ldhf suka")
            image = cv2.imread(path + "/" + file_name)
            res = [num]
            res += (list(get_features_from_image(image, get_linear_and_eccentricity_features)[0][0]))
            f.append(res)
    np.savetxt(dataset_path + "dataset.csv", f, delimiter=',', fmt='%f')


def create_classifier():
    dataset_path = "/home/alexander/GitHib/EmotionRecognition/Dataset/dataset.csv"
    dataset = loadtxt(open('Dataset/dataset.csv'), dtype='f8', delimiter=',')
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=4)
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    clf.fit(train, target)
    with open('classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
    f.close()


def main():
    image = cv2.imread("/home/ilya/Загрузки/images.jpg")
    res = list(get_features_from_image(image, get_linear_and_eccentricity_features)[0][0])
    with open('classifier.pkl', 'rb') as f:
        clf = pickle.load(f)
    f.close()
    result = clf.predict_proba([res])
    print(result)


def baluemsa():
    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        with open('classifier.pkl', 'rb') as f:
            clf = pickle.load(f)
            f.close()
        ret, frame = cap.read()
        a = get_features_from_image(frame, get_linear_and_eccentricity_features)
        for image in a:
            res = list(image[0])
            result = clf.predict_proba([res])
            pos1 = (image[1].left(), image[1].top())
            pos2= (image[1].right(), image[1].bottom())
            cv2.rectangle(frame, pos1,pos2,(255,0,0))
            cv2.putText(frame, str(result), pos1,cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0))
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    baluemsa()
