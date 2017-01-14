# show http://www.portointeractivecenter.org/site/wp-content/uploads/Real-Time-Emotion-Recognition_a-Novel-Method-for-Geometrical-Facial-Features-Extraction1.pdf

from geometric_helper import *
from image_processor import *
import openface

predictor_model = "shape_predictor_68_face_landmarks.dat"
face_aligner = openface.AlignDlib(predictor_model)


def get_features_from_image(image, get_features):
    features = []
    landmarks_of_faces = get_landmarks_and_positions_from_image(image)
    for landmarks, face_rect in landmarks_of_faces:
        feature = get_features(landmarks)
        features.append((feature, face_rect))

    return features


def get_linear_and_eccentricity_features_from_face(image, face_rect):
    landmarks = get_rotation_landmarks_from_face(image, face_rect)
    return get_linear_and_eccentricity_features(landmarks)


def preprocess_image(image):
    image = CLAHE(image)
    return get_gray_image(image)


def get_landmarks_and_positions_from_image(image):
    image = preprocess_image(image)
    landmarks_of_faces = []
    detected_faces = face_aligner.getAllFaceBoundingBoxes(image)
    for face_rect in detected_faces:
        landmarks = get_rotation_landmarks_from_face(image, face_rect)
        landmarks_of_faces.append((landmarks, face_rect))

    return landmarks_of_faces


def get_rotation_landmarks_from_face(image, face_rect):
    landmarks = face_aligner.findLandmarks(image, face_rect)
    center = get_center_between_eyes(landmarks)
    angle = get_rotation_angle(landmarks)
    rotation_matrix = get_rotation_matrix(center, angle)
    return get_rotation_points(landmarks, center, rotation_matrix)


# S1
def get_linear_features(landmarks):
    UEl_m3_y = get_y(UEl_m3(landmarks))
    SN_y = get_y(SN(landmarks))
    UEBl_m7_y = get_y(UEBl_m7(landmarks))
    U_m1_y = get_y(U_m1(landmarks))
    D_m2_y = get_y(D_m2(landmarks))

    DEN = distance_between_numbers(UEl_m3_y, SN_y)

    L1 = distance_between_numbers(UEBl_m7_y, UEl_m3_y) / DEN
    L2 = distance_between_numbers(U_m1_y, SN_y) / DEN
    L3 = distance_between_numbers(D_m2_y, SN_y) / DEN

    return L1, L2, L3


# S2
def get_eccentricity_features(landmarks):
    A_M_point = A_M(landmarks)
    B_M_point = B_M(landmarks)
    U_m1_point = U_m1(landmarks)
    D_m2_point = D_m2(landmarks)
    Ell_M_point = Ell_M(landmarks)
    Elr_M_point = Elr_M(landmarks)
    UEl_m3_point = UEl_m3(landmarks)
    DEl_m4_point = DEl_m4(landmarks)
    Erl_M_point = Erl_M(landmarks)
    Err_M_point = Err_M(landmarks)
    UEr_m5_point = UEr_m5(landmarks)
    DEr_m6_point = DEr_m6(landmarks)
    EBll_M_point = EBll_M(landmarks)
    EBlr_M_point = EBlr_M(landmarks)
    UEBl_m7_point = UEBl_m7(landmarks)
    EBrl_M_point = EBrl_M(landmarks)
    EBrr_M_point = EBrr_M(landmarks)
    UEBr_m8_point = UEBr_m8(landmarks)

    E1 = eccentricity_on_points(A_M_point, B_M_point, U_m1_point)
    E2 = eccentricity_on_points(A_M_point, B_M_point, D_m2_point)
    E3 = eccentricity_on_points(Ell_M_point, Elr_M_point, UEl_m3_point)
    E4 = eccentricity_on_points(Ell_M_point, Elr_M_point, DEl_m4_point)
    E5 = eccentricity_on_points(Erl_M_point, Err_M_point, UEr_m5_point)
    E6 = eccentricity_on_points(Erl_M_point, Err_M_point, DEr_m6_point)
    E7 = eccentricity_on_points(EBll_M_point, EBlr_M_point, UEBl_m7_point)
    E8 = eccentricity_on_points(EBrl_M_point, EBrr_M_point, UEBr_m8_point)

    return E1, E2, E3, E4, E5, E6, E7, E8


# S3
def get_linear_and_eccentricity_features(landmarks):
    L1, L2, L3 = get_linear_features(landmarks)
    E1, E2, E3, E4, E5, E6, E7, E8 = get_eccentricity_features(landmarks)

    return L1, L2, L3, E1, E2, E3, E4, E5, E6, E7, E8


def eccentricity_on_points(left_point, right_point, middle_point):
    a = distance_between_numbers(get_x(right_point), get_x(left_point)) / 2
    b = distance_between_numbers(get_y(left_point) - get_y(middle_point))

    return eccentricity(a, b)


def get_x(point):
    return point[0]


def get_y(point):
    return point[1]


def EBll_M(landmarks):
    return landmarks[17]


def UEBl_m7(landmarks):
    return landmarks[19]


def EBlr_M(landmarks):
    return landmarks[21]


def Ell_M(landmarks):
    return landmarks[36]


def UEl_m3(landmarks):
    return landmarks[37]


def Elr_M(landmarks):
    return landmarks[39]


def DEl_m4(landmarks):
    return landmarks[41]


def EBrl_M(landmarks):
    return landmarks[22]


def UEBr_m8(landmarks):
    return landmarks[24]


def EBrr_M(landmarks):
    return landmarks[26]


def Erl_M(landmarks):
    return landmarks[42]


def UEr_m5(landmarks):
    return landmarks[44]


def Err_M(landmarks):
    return landmarks[45]


def DEr_m6(landmarks):
    return landmarks[46]


def SN(landmarks):
    return landmarks[33]


def A_M(landmarks):
    return landmarks[48]


def U_m1(landmarks):
    return landmarks[51]


def B_M(landmarks):
    return landmarks[54]


def D_m2(landmarks):
    return landmarks[57]


def get_rotation_angle(landmarks_pos):
    return get_angle(landmarks_pos[36], landmarks_pos[45])


def get_center_between_eyes(landmarks_pos):
    return get_center(landmarks_pos[36], landmarks_pos[45])
