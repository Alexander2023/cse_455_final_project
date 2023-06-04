from cv2 import VideoCapture, imshow, waitKey, putText, FONT_HERSHEY_SIMPLEX, cvtColor, COLOR_BGR2RGB
from PIL import Image
from random import randint

HAPPY_LABEL = "happy"
INFERENCE_RATE = 10

labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def load_model():
    pass

def generate_expression_label(video_frame):
    index = randint(0, 6)
    return labels[index]

def save_happy_image(video_frame):
    video_frame = cvtColor(video_frame, COLOR_BGR2RGB)
    image = Image.fromarray(video_frame)
    image.save("happy_image.jpg")

camera_feed = VideoCapture(0)
load_model()

i = 0
current_label = ""
try:
    while True:
        has_received_frame, video_frame = camera_feed.read()

        if has_received_frame:
            if i % INFERENCE_RATE == 0:
                current_label = generate_expression_label(video_frame)
                i = 0

            putText(video_frame, current_label, (50, 50), FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 0))
            imshow('video_frame', video_frame)
            waitKey(25)

            if current_label == HAPPY_LABEL:
                save_happy_image(video_frame)

        i += 1
finally:
    camera_feed.release()
