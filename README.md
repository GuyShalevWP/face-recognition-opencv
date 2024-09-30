# Simple Face Recognition App

This is a simple face recognition app built using **OpenCV** for capturing camera input and **Kivy** for the GUI. The app uses **Mediapipe's Face Mesh** to capture and compare facial landmarks for face registration and login.

## Features

-   **Face Registration**: The app prompts the user to look straight, left, and right, capturing their face from three different angles.
-   **Face Login**: The app compares the current face with the previously registered face data to authenticate the user.
-   **Visual Feedback**: The camera border turns green on a successful login or red if the face is not recognized.
-   **Sound Prompts**: The app plays a sound to prompt the user to turn their head during registration and a different sound when the registration is successfully completed.

## Try it yourself

1. Clone the Repository:

```
git clone https://github.com/GuyShalevWP/face-recognition-opencv.git
cd project
```

2. Create a Virtual Environment:

```
python -m virtualenv env
source env\Scripts\activate
```

3. Install Dependencies:

```
pip install -r requirements.txt
```

4. Run the app:

```
python app.py
```
