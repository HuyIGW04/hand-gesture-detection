import os
import cv2
import time
import yaml
import torch
import numpy as np
try:
    from controller import ModbusMaster as _HardwareModbus
except Exception:
    _HardwareModbus = None
from torch import nn
import mediapipe as mp
from collect_hand import HandDetector
from collect_hand import get_label_from_config
NUM_CLASSES = len(get_label_from_config('../config.yaml'))


class MockController:
    """Simple stand-in for the hardware controller for local development."""

    def __init__(self):
        self._state = {1: False, 2: False, 3: False}

    def _log(self, idx, state):
        status = "ON" if state else "OFF"
        print(f"[MockController] actuator_{idx} -> {status}")

    def switch_actuator_1(self, state):
        state = bool(state)
        self._state[1] = state
        self._log(1, state)

    def switch_actuator_2(self, state):
        state = bool(state)
        self._state[2] = state
        self._log(2, state)

    def switch_actuator_3(self, state):
        state = bool(state)
        self._state[3] = state
        self._log(3, state)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        list_label = NUM_CLASSES
        self.linear_stack = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.6)
        )
        self.output = nn.Linear(128, list_label)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_stack(x)
        x = self.output(x)
        return x
    
    # updating
    def predict(self,x,threshold=0.9):
        logits = self(x)
        softmax_prob = nn.Softmax(dim=1)(logits)
        chosen_ind = torch.argmax(softmax_prob,dim=1)
        return torch.where(softmax_prob[0,chosen_ind]>threshold,chosen_ind,-1)
    
    def get_pred_label(self, x):
        """return predicted class vector"""
        logits = self(x)
        out = nn.Softmax(dim=1)(logits)
        return torch.argmax(out, dim=1)


class LightGesture:
    def __init__(self, model_path, device=False):
        self.device = False
        self.height = 720
        self.width = 1280

        self.detector = HandDetector()
        self.status_text = None
        self.signs = get_label_from_config("../config.yaml")
        self.classifier = MLP()
        self.classifier.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.classifier.eval()

        self.controller = MockController()
        if device:
            try:
                if _HardwareModbus is None:
                    raise RuntimeError("Hardware controller is unavailable.")
                self.controller = _HardwareModbus()
                self.device = True
            except Exception as exc:
                print(f"[MockController] Falling back to simulation: {exc}")
        self.light1 = False
        self.light2 = False
        self.light3 = False
    

    def light_device(self, img, lights):
        # Append a white rectangle at the bottom of the image
        height, width, _ = img.shape
        rect_height = int(0.15 * height)
        rect_width = width
        white_rect = np.ones((rect_height, rect_width, 3), dtype=np.uint8) * 255

        # Draw a red border around the rectangle
        cv2.rectangle(white_rect, (0, 0), (rect_width, rect_height), (0, 0, 255), 2)

        # Calculate circle positions
        circle_radius = int(0.45*rect_height)
        circle1_center = (int(rect_width * 0.25), int(rect_height / 2))
        circle2_center = (int(rect_width * 0.5), int(rect_height / 2))
        circle3_center = (int(rect_width * 0.75), int(rect_height / 2))

        # Draw the circles
        on_color = (0, 255, 255)
        off_color = (0, 0, 0)
        colors = [off_color, on_color]
        circle_centers = [circle1_center, circle2_center, circle3_center]
        for cc, light in zip(circle_centers, lights):
            color = colors[int(light)]
            cv2.circle(white_rect, cc, circle_radius, color, -1)

        # Append the white rectangle to the bottom of the image
        img = np.vstack((img, white_rect))
        return img

    def run(self):
        cam =  cv2.VideoCapture(0)
        cam.set(3,1280)
        cam.set(4,720)
        while cam.isOpened():
            _,frame = cam.read()

            hand,img = self.detector.detect(frame)
            if len(hand) != 0:
                with torch.no_grad():
                    hand_landmark = torch.from_numpy(np.array(hand[0],dtype=np.float32).flatten()).unsqueeze(0)
                    class_number = self.classifier.predict(hand_landmark).item()
                    if class_number != -1:
                        self.status_text = self.signs[class_number]

                        if self.status_text == "light1":
                            if self.light1 is False:
                                print("lights on")
                                self.light1 = True
                                self.controller.switch_actuator_1(True)
                        elif self.status_text == "light2":
                            if self.light2 is False:
                                self.light2 = True
                                self.controller.switch_actuator_2(True)
                        elif self.status_text == "light3":
                            if self.light3 is False:
                                self.light3 = True
                                self.controller.switch_actuator_3(True)
                        elif self.status_text == "turn_on":
                            previous_state = (self.light1, self.light2, self.light3)
                            self.light1 = self.light2 = self.light3 = True
                            if not all(previous_state):
                                self.controller.switch_actuator_1(self.light1)
                                if self.device:
                                    time.sleep(0.03)
                                self.controller.switch_actuator_2(self.light2)
                                if self.device:
                                    time.sleep(0.03)
                                self.controller.switch_actuator_3(self.light3)
                        elif self.status_text == "turn_off":
                            previous_state = (self.light1, self.light2, self.light3)
                            if any(previous_state):
                                self.light1 = self.light2 = self.light3 = False
                                self.controller.switch_actuator_1(self.light1)
                                if self.device:
                                    time.sleep(0.03)
                                self.controller.switch_actuator_2(self.light2)
                                if self.device:
                                    time.sleep(0.03)
                                self.controller.switch_actuator_3(self.light3)
                                
                    else:
                        self.status_text = "undefined command"
                        
            else:
                self.status_text = None

            img = self.light_device(img, [self.light1, self.light2, self.light3])

            display_text = self.status_text if self.status_text is not None else ""
            cv2.putText(img, display_text, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.namedWindow('window', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('window', 1920, 1080)
            cv2.imshow("window",img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        cv2.destroyAllWindows()        



if __name__ == "__main__":
    model_path = "./models/model_28-10_15-46_MLP_best"
    use_device = os.environ.get("USE_LIGHT_DEVICE", "").lower() in {"1", "true", "yes"}
    light = LightGesture(model_path, device=use_device)
    light.run()
