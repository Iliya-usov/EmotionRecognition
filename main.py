from feature_detector import *
from sklearn.ensemble import RandomForestClassifier
import pickle


def create_classifier():
    dataset = np.loadtxt(open('Dataset/dataset.csv'), dtype='f8', delimiter=',')
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=4)
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    clf.fit(train, target)
    with open('classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
    f.close()


def get_emotions(emotions):
    return \
        "neutral = {}\n" \
        "anger = {}\n" \
        "contempt = {}\n" \
        "disgust = {}\n" \
        "fear = {}\n" \
        "happy = {}\n" \
        "sadness = {}\n" \
        "surprise = {}".format(
            emotions[0],
            emotions[1],
            emotions[2],
            emotions[3],
            emotions[4],
            emotions[5],
            emotions[6],
            emotions[7])


def main():
    image = cv2.imread("/home/ilya/Загрузки/уд.jpg")
    a = get_features_from_image(image, get_linear_and_eccentricity_features)
    with open('classifier.pkl', 'rb') as f:
        clf = pickle.load(f)
    f.close()
    for img in a:
        res = list(img[0])
        result = clf.predict_proba([res])
        pos1 = (img[1].left(), img[1].top())
        pos2 = (img[1].right(), img[1].bottom())
        cv2.rectangle(image, pos1, pos2, (255, 0, 0))
        em = str.split(get_emotions(result[0]),'\n')
        pos1 = (0, 430)
        for e in em:
            pos1 = (pos1[0], pos1[1] + 20)
            cv2.putText(image, e, pos1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        print(get_emotions(result[0]))

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
