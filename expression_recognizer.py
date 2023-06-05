import torch
from cv2 import VideoCapture, imshow, waitKey, putText, FONT_HERSHEY_SIMPLEX, cvtColor, COLOR_BGR2RGB, COLOR_BGR2GRAY
from PIL import Image
from torchvision import models, transforms
from torch import load, no_grad, nn, argmax, unsqueeze, device

from learning import SAVED_WEIGHTS_PATH, CLASSES, MODEL_FORMAT_TRANSFORMS

HAPPY_LABEL = "happy"
INFERENCE_RATE = 5
LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def load_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, CLASSES)
    state_dict = load(SAVED_WEIGHTS_PATH, device("cpu"))
    model.load_state_dict(state_dict)
    return model

def generate_expression_label(video_frame, model_format_transform):
    # matches color space during training
    video_frame = cvtColor(video_frame, COLOR_BGR2GRAY)

    image = Image.fromarray(video_frame)

    with no_grad():
      image = model_format_transform(image)
      image = unsqueeze(image, 0) # needed since batch size is 1
      prediction = model(image)
      label_index = argmax(prediction).item()
      return LABELS[label_index]

def save_happy_image(video_frame):
    video_frame = cvtColor(video_frame, COLOR_BGR2RGB)
    image = Image.fromarray(video_frame)
    image.save("happy_image.jpg")

model_format_transform = transforms.Compose(MODEL_FORMAT_TRANSFORMS)
camera_feed = VideoCapture(0)
model = load_model()
model.eval()

i = 0
current_label = ""
is_already_happy = False
try:
    while True:
        has_received_frame, video_frame = camera_feed.read()

        if has_received_frame:
            if i % INFERENCE_RATE == 0:
                current_label = generate_expression_label(video_frame, model_format_transform)
                i = 0

            putText(video_frame, current_label, (50, 50), FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 0))
            imshow('video_frame', video_frame)
            waitKey(25)

            if current_label == HAPPY_LABEL and not is_already_happy:
                save_happy_image(video_frame)
                is_already_happy = True
            elif current_label != HAPPY_LABEL:
                is_already_happy = False

        i += 1
finally:
    camera_feed.release()
