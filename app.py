import cv2
import mediapipe as mp
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
import numpy as np
from kivy.core.audio import SoundLoader  # Using Kivy's SoundLoader for cross-platform

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# To store the registered face landmarks for front, left, and right views
registered_faces = {
    "front": None,
    "left": None,
    "right": None
}

def play_beep():
    """Play beep sound to prompt the user to turn their head."""
    beep_sound = SoundLoader.load('confirm_sound.mp3')  # Load the sound file for prompts
    if beep_sound:
        beep_sound.play()

def play_success_sound():
    """Play success sound when registration is complete."""
    success_sound = SoundLoader.load('success_sound.mp3')  # Load the success sound file
    if success_sound:
        success_sound.play()

class CameraWidget(Image):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_color = (1, 1, 1, 1)  # Default white border

    def set_border_color(self, r, g, b, a=1):
        self.border_color = (r, g, b, a)
        self.update_border()

    def update_border(self):
        with self.canvas.before:
            self.canvas.before.clear()
            Color(*self.border_color)
            Rectangle(pos=self.pos, size=self.size)

class FaceRecognitionApp(App):
    def build(self):
        # Main layout
        self.layout = BoxLayout(orientation='vertical')

        # Camera feed widget
        self.camera_widget = CameraWidget(size_hint=(1, 0.6))  # 60% of height
        self.layout.add_widget(self.camera_widget)

        # Message label for feedback
        self.message_label = Label(text="No face registered yet", font_size=24, size_hint=(1, 0.1))
        self.layout.add_widget(self.message_label)

        # Button layout
        button_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))

        # Register button
        self.register_button = Button(text="Register Face", font_size=24)
        self.register_button.bind(on_press=self.start_face_registration)
        button_layout.add_widget(self.register_button)

        # Login button
        self.login_button = Button(text="Login", font_size=24)
        self.login_button.bind(on_press=self.login)
        button_layout.add_widget(self.login_button)

        self.layout.add_widget(button_layout)

        # Open camera
        self.capture = cv2.VideoCapture(0)

        # Schedule the camera update (30 FPS)
        Clock.schedule_interval(self.update_camera, 1.0 / 30.0)

        return self.layout

    def update_camera(self, dt):
        # Read frame from the camera
        ret, frame = self.capture.read()
        if ret:
            # Convert it to a Kivy texture
            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.camera_widget.texture = texture

    def detect_face_landmarks(self, frame):
        # Convert the image to RGB format (Mediapipe expects RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect face landmarks
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]  # Return the first face's landmarks
        return None

    def start_face_registration(self, instance):
        # Start the face registration process, asking the user to look in different directions
        self.message_label.text = "Please look straight at the camera"
        Clock.schedule_once(self.capture_front_face, 3)  # Allow more time for the user to turn

    def capture_front_face(self, dt):
        global registered_faces
        ret, frame = self.capture.read()
        if ret:
            # Detect face landmarks in the frame
            landmarks = self.detect_face_landmarks(frame)
            if landmarks:
                registered_faces['front'] = landmarks
                self.message_label.text = "Now look to your left (hold still)"
                play_beep()  # Play beep when asking to turn
                Clock.schedule_once(self.capture_left_face, 4)  # Extra time for user to turn

    def capture_left_face(self, dt):
        global registered_faces
        ret, frame = self.capture.read()
        if ret:
            # Detect face landmarks in the frame
            self.message_label.text = "Capturing face, please hold still..."
            for i in range(10):  # Capture multiple frames for better accuracy
                ret, frame = self.capture.read()
                landmarks = self.detect_face_landmarks(frame)
                if landmarks:
                    registered_faces['left'] = landmarks
                    break
            self.message_label.text = "Now look to your right (hold still)"
            play_beep()  # Play beep when asking to turn right
            Clock.schedule_once(self.capture_right_face, 4)

    def capture_right_face(self, dt):
        global registered_faces
        ret, frame = self.capture.read()
        if ret:
            # Detect face landmarks in the frame
            self.message_label.text = "Capturing face, please hold still..."
            for i in range(10):  # Capture multiple frames for better accuracy
                ret, frame = self.capture.read()
                landmarks = self.detect_face_landmarks(frame)
                if landmarks:
                    registered_faces['right'] = landmarks
                    break
            if registered_faces['right']:
                self.message_label.text = "Face registration complete!"
                play_success_sound()  # Play success sound when registration is complete
            else:
                self.message_label.text = "Failed to detect face on the right. Try again."

    def login(self, instance):
        global registered_faces

        if not all(registered_faces.values()):
            self.message_label.text = "Please register your face first"
            return

        ret, frame = self.capture.read()
        if ret:
            # Detect face landmarks in the frame
            current_landmarks = self.detect_face_landmarks(frame)
            if current_landmarks:
                # Compare the detected landmarks with the registered faces
                if (self.compare_faces(registered_faces['front'], current_landmarks) or
                        self.compare_faces(registered_faces['left'], current_landmarks) or
                        self.compare_faces(registered_faces['right'], current_landmarks)):
                    self.message_label.text = "Logged In"
                    self.camera_widget.set_border_color(0, 1, 0)  # Green border on success
                else:
                    self.message_label.text = "Face not recognized"
                    self.camera_widget.set_border_color(1, 0, 0)  # Red border on fail
            else:
                # If no face is detected at all
                self.message_label.text = "No face detected"
                self.camera_widget.set_border_color(1, 0, 0)  # Red border if no face

    def compare_faces(self, registered_landmarks, current_landmarks):
        # Compare the two sets of facial landmarks using Euclidean distance
        registered_landmark_array = np.array([(lm.x, lm.y, lm.z) for lm in registered_landmarks.landmark])
        current_landmark_array = np.array([(lm.x, lm.y, lm.z) for lm in current_landmarks.landmark])

        # Calculate the mean Euclidean distance between the two sets of landmarks
        distance = np.linalg.norm(registered_landmark_array - current_landmark_array, axis=1).mean()

        # If the distance is small enough, we assume it's the same person
        threshold = 0.05  # Adjust the threshold for sensitivity
        return distance < threshold

    def on_stop(self):
        # Release the camera when the app is closed
        self.capture.release()

if __name__ == "__main__":
    FaceRecognitionApp().run()
