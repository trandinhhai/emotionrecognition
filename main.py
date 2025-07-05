from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

# Load face detector and trained emotion classifier model
face_classifier = cv2.CascadeClassifier(r'D:\Download_D\emotionrecognition-main\haarcascade_frontalface_default.xml')
classifier = load_model(r'D:\Download_D\emotionrecognition-main\Emotion_little_vgg.h5')

# Ensure class labels match the model's training set
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Open webcam
cap = cv2.VideoCapture(0)   

#Process video bos
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum(roi_gray) != 0:
            # Normalize input image
            roi = roi_gray.astype("float") / 255.0  
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=-1)  # Add grayscale channel
            roi = np.expand_dims(roi, axis=0)  # Add batch dimension

            # Handle model output
            preds = classifier.predict(roi)
            if preds.shape[1] != len(class_labels):
                print("Warning: Model prediction shape mismatch with class labels!")
                continue

            # Get highest probability emotion
            confidence = np.max(preds) * 100  
            label = f"{class_labels[np.argmax(preds)]}: {confidence:.2f}%"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Face Found", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Facial Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()                                                                                            