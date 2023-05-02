import cv2, mmcv
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
import os
from tkinter.messagebox import showerror
from datetime import datetime

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch


class FaceRecognitionInterface(tk.Tk):
    BACKGROUND_IMAGES_DIR = 'background_images'
    BACKGROUND_VIDEOS_DIR = "background_videos"
    FACES_DIR = 'faces'
    NO_BACKGROUND_TEXT = 'Нет'
    IMAGE_RESIZE_FACTOR = 4
    IMAGE_SIZE = (800, 600)
    FACE_MATCH_TRESHOLD = 0.9

    def __init__(self):
        super().__init__()

        self.title('Face recognition')

        self.mirror = tk.IntVar(self)
        self.fps = tk.IntVar(self, 60)
        self.background_image = tk.StringVar(self, self.NO_BACKGROUND_TEXT)
        self.background_video = tk.StringVar(self, self.NO_BACKGROUND_TEXT)
        self.background_video.trace_add("write", self.on_video_selected)
        self.identification = tk.IntVar(self, 1)
        self.font = ImageFont.truetype('Roboto-Regular.ttf', 16)
        self.frame_number = 0
        self.last_frame_datetime = None

        self.background_image_names = os.listdir(self.BACKGROUND_IMAGES_DIR)
        background_image_paths = [os.path.join(self.BACKGROUND_IMAGES_DIR, name) for name in self.background_image_names]
        self.background_images = [cv2.imread(path) for path in background_image_paths]

        self.background_video_names = os.listdir(self.BACKGROUND_VIDEOS_DIR)
        background_video_paths = [os.path.join(self.BACKGROUND_VIDEOS_DIR, name) for name in self.background_video_names]
        self.background_videos = []
        for video_path in background_video_paths:
            video = mmcv.VideoReader(video_path)
            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in video]
            self.background_videos.append(frames)

        settings_frame = tk.Frame(self)
        tk.Label(settings_frame, text='Идентификация: ').pack(side='left')
        tk.Checkbutton(settings_frame, variable=self.identification).pack(side='left', padx=(0, 16))
        tk.Label(settings_frame, text='Частота обработки (кадров в секунду): ').pack(side='left')
        tk.Spinbox(settings_frame, textvariable=self.fps, from_=1, to=100).pack(side='left', padx=(0, 16))
        tk.Label(settings_frame, text='Фоновое изображение').pack(side='left')
        tk.OptionMenu(settings_frame, self.background_image, *([self.NO_BACKGROUND_TEXT] + self.background_image_names)).pack(side='left')
        tk.Label(settings_frame, text='Фоновое видео').pack(side='left')
        tk.OptionMenu(settings_frame, self.background_video, *([self.NO_BACKGROUND_TEXT] + self.background_video_names)).pack(
            side='left')
        settings_frame.pack()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running on device: {device}")
        mtcnn = MTCNN(device=device)
        resnet = InceptionResnetV1(pretrained="vggface2").eval()

        self.face_embeddings = {}
        for name in os.listdir(self.FACES_DIR):
            self.face_embeddings[name] = []
            for image_path in os.listdir(os.path.join(self.FACES_DIR, name)):
                image = cv2.imread(os.path.join(self.FACES_DIR, name, image_path))
                faces, prob = mtcnn.detect(image)
                if len(faces) < 1:
                    showerror('Ошибка', f'На изображении {os.path.join(name, image_path)} '
                                        f'не найдены лица.')
                    self.destroy()
                    return
                elif len(faces) > 1:
                    showerror('Ошибка', f'На изображении {os.path.join(name, image_path)} '
                                        f'найдено больше одного лица.')
                    self.destroy()
                    return
                x0, y0, x1, y1 = [int(point) for point in faces[0]]
                face_image = image[y0:y1, x0:x1]
                tensor = mtcnn(face_image)
                self.face_embeddings[name].append(resnet(tensor.unsqueeze(0)))

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap is None or not cap.isOpened():
            showerror('Ошибка', 'Не удалось подключить камеру')
            self.destroy()
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        #
        video_label = tk.Label(self)
        video_label.pack(fill='both', expand=1)

        def show_frame():
            elapsed_seconds = (datetime.now() - self.last_frame_datetime).total_seconds()
            self.last_frame_datetime = datetime.now()

            _, camera_image = cap.read()
            camera_image = cv2.flip(camera_image, 1)
            camera_image = cv2.resize(camera_image, self.IMAGE_SIZE)
            if self.background_video.get() != self.NO_BACKGROUND_TEXT:
                background_video = self.background_videos[self.background_video_names.index(self.background_video.get())]
                background_image = None
            elif self.background_image.get() != self.NO_BACKGROUND_TEXT:
                background_image = self.background_images[self.background_image_names.index(self.background_image.get())]
                background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
                background_video = None
            else:
                background_video = None
                background_image = None
            camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)
            small_camera_image = cv2.resize(camera_image, (0, 0), fx=1/self.IMAGE_RESIZE_FACTOR, fy=1/self.IMAGE_RESIZE_FACTOR)

            face_locations, prob = mtcnn.detect(small_camera_image)
            if face_locations is not None and self.identification.get():
                face_embeddings = []
                for face in face_locations:
                    x0, y0, x1, y1 = [int(point) for point in face]
                    face_image = small_camera_image[y0:y1, x0:x1]
                    if face_image.shape[0] < 2 or face_image.shape[1] < 2:
                        continue
                    try:
                        tensor = mtcnn(face_image)
                        if tensor is not None:
                            face_embeddings.append(resnet(tensor.unsqueeze(0)))
                    except RuntimeError as error:
                        print(error)

            if background_video is not None:
                self.frame_number += int(self.fps.get() * elapsed_seconds)
                current_frame = background_video[self.frame_number % len(background_video)]
                result_image = cv2.resize(current_frame, self.IMAGE_SIZE)
                if face_locations is not None:
                    for i, face in enumerate(face_locations):
                        x0, y0, x1, y1 = [int(coord) * self.IMAGE_RESIZE_FACTOR for coord in face]
                        result_image[y0:y1, x0:x1] = camera_image[y0:y1, x0:x1]
            elif background_image is not None:
                result_image = cv2.resize(background_image, self.IMAGE_SIZE)
                if face_locations is not None:
                    for i, face in enumerate(face_locations):
                        x0, y0, x1, y1 = [int(coord) * self.IMAGE_RESIZE_FACTOR for coord in face]
                        result_image[y0:y1, x0:x1] = camera_image[y0:y1, x0:x1]
            else:
                result_image = camera_image

            pil_image = Image.fromarray(result_image)
            image_draw = ImageDraw.Draw(pil_image)
            if face_locations is not None:
                for i, face in enumerate(face_locations):
                    x0, y0, x1, y1 = [int(coord * self.IMAGE_RESIZE_FACTOR) for coord in face]

                    image_draw.rectangle([(x0, y0), (x1, y1)], outline='red', width=2)
                    if self.identification.get():
                        name = 'Неизвестный'
                        if len(face_locations) == len(face_embeddings):
                            for known_name in self.face_embeddings:
                                for face_embedding in self.face_embeddings[known_name]:
                                    result = (face_embedding - face_embeddings[i]).norm().item()
                                    print(f"{known_name}: {result}")
                                    if result <= self.FACE_MATCH_TRESHOLD:
                                        name = known_name
                        image_draw.rectangle([(x0, y1), (x1, y1 + 40)], fill='red')
                        image_draw.text((x0 + 6, y1 + 20), name, font=self.font)

            imgtk = ImageTk.PhotoImage(image=pil_image)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
            video_label.after(1000 // self.fps.get(), show_frame)

        self.last_frame_datetime = datetime.now()
        show_frame()

    def on_video_selected(self, *args):
        self.frame_number = 0

if __name__ == '__main__':
    FaceRecognitionInterface().mainloop()

