import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
import mediapipe as mp

class DigitSequenceRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Handwritten Digit Sequence Recognition")
        
        # Set up the GUI
        self.canvas = tk.Canvas(self.master, width=640, height=480)
        self.canvas.pack(pady=20)
        
        self.result_label = tk.Label(self.master, text="Prediction: ", font=("Arial", 24))
        self.result_label.pack(pady=10)
        
        self.clear_button = tk.Button(self.master, text="Clear", command=self.clear_sequence)
        self.clear_button.pack(pady=10)
        
        # Load the trained model
        self.model = self.load_model()
        
        # Set up camera and hand detection
        self.camera = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Set up variables for finger tracking
        self.prev_points = []
        self.current_stroke = []
        self.all_strokes = []
        self.is_drawing = False
        self.draw_color = (255, 255, 255)  # White color for drawing
        self.draw_thickness = 15
        
        self.digit_sequence = []
        self.max_sequence_length = 5
        
        # Create a separate canvas for displaying drawn digits
        self.digit_canvas = tk.Canvas(self.master, width=280, height=56)
        self.digit_canvas.pack(pady=10)
        
        self.update_camera()

    def load_model(self):
        return tf.keras.models.load_model('mnist_sequence_model.h5')

    def predict_digit(self, image):
        # Preprocess the image
        img = cv2.resize(image, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        img = img.reshape(1, 28, 28, 1) / 255.0
        
        # Add the digit to the sequence
        self.digit_sequence.append(img)
        
        # Update the digit canvas
        self.update_digit_canvas()
        
        # If we have enough digits, make a prediction
        if len(self.digit_sequence) == self.max_sequence_length:
            sequence = np.array(self.digit_sequence)
            prediction = self.model.predict(sequence.reshape(1, self.max_sequence_length, 28, 28, 1))
            digits = np.argmax(prediction, axis=-1)[0]
            
            result = ''.join(map(str, digits))
            self.result_label.config(text=f"Prediction: {result}")
        else:
            self.result_label.config(text=f"Drawn: {len(self.digit_sequence)} of {self.max_sequence_length}")

    def clear_sequence(self):
        self.digit_sequence = []
        self.result_label.config(text="Prediction: ")
        self.digit_canvas.delete("all")

    def update_digit_canvas(self):
        self.digit_canvas.delete("all")
        for i, digit_img in enumerate(self.digit_sequence):
            img = Image.fromarray((digit_img.reshape(28, 28) * 255).astype('uint8'))
            img = ImageTk.PhotoImage(img)
            self.digit_canvas.create_image(i*56, 0, anchor=tk.NW, image=img)
            self.digit_canvas.image = img

    def update_camera(self):
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect hands
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Get index finger tip and middle finger tip coordinates
                    index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    
                    x, y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
                    
                    # Check if index finger is raised and middle finger is lowered (drawing gesture)
                    if index_finger_tip.y < middle_finger_tip.y:
                        self.is_drawing = True
                        # Draw the finger trajectory
                        if self.prev_points:
                            cv2.line(frame, self.prev_points[-1], (x, y), self.draw_color, self.draw_thickness)
                            self.current_stroke.append((x, y))
                        self.prev_points.append((x, y))
                    else:
                        if self.is_drawing:
                            self.is_drawing = False
                            if self.current_stroke:
                                self.all_strokes.append(self.current_stroke)
                                self.current_stroke = []
                            
                            # If we have collected strokes, try to recognize the digit
                            if self.all_strokes:
                                digit_image = self.create_digit_image(frame.shape[1], frame.shape[0])
                                self.predict_digit(digit_image)
                                self.all_strokes = []
                        self.prev_points = []
            
            # Convert the frame to ImageTk format
            photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.image = photo
        
        self.master.after(10, self.update_camera)

    def create_digit_image(self, width, height):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for stroke in self.all_strokes:
            for i in range(len(stroke) - 1):
                cv2.line(image, stroke[i], stroke[i+1], self.draw_color, self.draw_thickness)
        return image

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitSequenceRecognitionApp(root)
    root.mainloop()
