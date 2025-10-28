import os
import cv2
import csv
import yaml
import numpy as np
import mediapipe as mp


def char_valid_checking(i: str):
    i = i.lower()
    return ord(i) == ord(' ') or (ord(i) < ord('q') and ord(i) >= ord('a'))

# {class: <action>}


def get_label_from_config(PATH: str) -> dict:
    with open(PATH, 'r') as f:
        label_dict = yaml.full_load(f)["gestures"]
    return label_dict


CONFIG_PATH = './config.yaml'
print(get_label_from_config(CONFIG_PATH))


# check axis
class CSVWriterBot:
    def __init__(self, PATH):
        self.csv_file = open(PATH, "a", newline="")
        self.writer = csv.writer(
            self.csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")

    def add_row(self, hand_gestures, label):
        self.writer.writerow(
            [label, *np.array(hand_gestures).flatten().tolist()])   # unlist

    def close(self):
        self.csv_file.close()


# important class: detect
class HandDetector:
    def __init__(self):

        self.mp_hands = mp.solutions.hands
        self.hand_drawing = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles
        self.detector = self.mp_hands.Hands(
            False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

    def detect(self, frame):
        hands_position = []
        frame = cv2.flip(frame, 1)
        background = frame.copy()
        cvt_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_collection = self.detector.process(cvt_frame)
        if hands_collection.multi_hand_landmarks is None:
            return [], background
        else:
            for hand in hands_collection.multi_hand_landmarks:
                hand_position = []
                self.hand_drawing.draw_landmarks(
                    background,
                    hand,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.styles.get_default_hand_landmarks_style(),
                    self.styles.get_default_hand_connections_style()
                )

                for landmark in hand.landmark:
                    x, y, z = landmark.x, landmark.y, landmark.z
                    hand_position = hand_position + [x, y, z]
                hands_position.append(hand_position)
            return hands_position, background


def run(DATA_PATH, SIGN_IMG_PATH, split="train", resolution=(1280, 720)):
    # data setup
    os.makedirs(DATA_PATH, exist_ok=True)       # csv
    os.makedirs(SIGN_IMG_PATH, exist_ok=True)   # image

    # train, val, test
    SPLIT_PATH = f'./{DATA_PATH}/{split}_split.csv'

    # object
    writer = CSVWriterBot(SPLIT_PATH)
    detector = HandDetector()

    # turn on webcam
    capture = cv2.VideoCapture(0)
    capture.set(3, resolution[0])
    capture.set(4, resolution[1])

    current_letter = None
    cannot_switch = False
    status = None
    save_frame = None
    label = None

    # start app
    while capture.isOpened():
        _, frame = capture.read()
        hand_list, background = detector.detect(frame)

        if current_letter is None:
            status = 'Press any key to record'
        else:  #
            if (ord('a') - ord(current_letter)) == 65:
                label = -1
                status = 'Recording unknown, press space again to stop'
            else:
                label = ord(current_letter) - ord('a')
                status = f'Recording {LABEL_CONFIG[label]}, press {current_letter} to stop'

        key = cv2.waitKey(1)
        if (key == -1):
            if (current_letter is None):
                pass
            else:
                if len(hand_list) != 0:
                    hand = hand_list[0]
                    writer.add_row(hand, label)
                    save_frame = frame

        else:
            key = chr(key)
            if key == 'q':   # q -> close
                break
            if char_valid_checking(key):        # valid key
                if (current_letter is None):
                    current_letter = key
                    cannot_switch = False
                elif (current_letter == key):
                    if save_frame is not None:
                        if label >= 0:
                            cv2.imwrite(
                                f"./{SIGN_IMG_PATH}/{LABEL_CONFIG[label]}.jpg", save_frame)

                    current_letter = None
                    cannot_switch = False
                    save_frame = None
                else:
                    cannot_switch = True

            else:
                if (current_letter is None):
                    cv2.putText(background, "please press a valid character", (0, 450),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    continue
                else:
                    cannot_switch = True
            if (cannot_switch):
                cv2.putText(background, f"please press {current_letter} again to unbind", (
                    0, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(background, status, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(f"{split}", background)
    cv2.destroyAllWindows()


# main function
if __name__ == "__main__":
    LABEL_CONFIG = get_label_from_config('config.yaml')
    data_path = './sign_data'
    sign_img_path = './sign_img'
    run(data_path, sign_img_path, "train", (1280, 720))
    run(data_path, sign_img_path, "val", (1280, 720))
    run(data_path, sign_img_path, "test", (1280, 720))
