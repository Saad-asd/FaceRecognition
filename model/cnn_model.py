import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
#
# img = cv2.imread('./test_images/messi1.jpg')
# # print(type(img))
# # print(img.shape)
# # plt.imshow(img)
# # plt.show()
#
# # print(img.shape)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # print(gray.shape)
#
# # plt.imshow(gray, cmap='gray')
# # plt.show()
#
#haar_cascade works with sliding-window approach
face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')
# #scalefactor = 1.3
# #minNeighbours = 5
# # keep min neighbours high to reduce false pos
# faces = face_cascade.detectMultiScale(gray, 1.3, 10)
# # print(faces)
# # print(faces[0])
# # print(faces[0][0])
#
#
# # (x,y,w,h) = faces[0]
# # print(x,y,w,h)
# # # cv2.rectangle(cv2Image,(x,y),(x+w,y+h),rgb value, thickness)
# # face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# # plt.imshow(face_img)
# # plt.show()
#
#
def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
#
#
# # cv2.destroyAllWindows()
# # for (x, y, w, h) in faces:
# #     face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
# #     roi_gray = gray[y:y + h, x:x + w]
# #     roi_color = face_img[y:y + h, x:x + w]
# #     eyes = eye_cascade.detectMultiScale(roi_color)
# #     for (ex, ey, ew, eh) in eyes:
# #         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
# #
# # plt.figure()
# # plt.imshow(face_img, cmap='gray')
# # plt.show()
#
path_to_data = "./data/"
path_to_cropped_data = "./cropped_data/"
#
import os
img_directories = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_directories.append(entry.path)
print(img_directories)

import shutil
if os.path.exists(path_to_cropped_data):
     shutil.rmtree(path_to_cropped_data)
os.mkdir(path_to_cropped_data)

cropped_image_directories = []
file_names_dict = {}
for img_dir in img_directories:
    count = 1
    name = img_dir.split('/')[-1]
    file_names_dict[name] = []
    for entry in os.scandir(img_dir):
        print(entry.path)
        roi_color = get_cropped_image_if_2_eyes(entry.path)
        if roi_color is not None:
            cropped_folder = path_to_cropped_data + name
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_directories.append(cropped_folder)
                print("Generating cropped images in folder: ",cropped_folder)
            cropped_file_name = name + str(count) + ".png"
            cropped_file_path = cropped_folder + "/" + cropped_file_name
            cv2.imwrite(cropped_file_path, roi_color)
            file_names_dict[name].append(cropped_file_path)
            count += 1
        else:
            pass

person_file_names_dict = {}
for img_dir in cropped_image_directories:
    name = img_dir.split('/')[-1]
    file_list = []
    for entry in os.scandir(img_dir):
        file_list.append(entry.path)
    person_file_names_dict[name] = file_list
print(person_file_names_dict)

class_dict = {}
count = 0
for name in person_file_names_dict.keys():
    class_dict[name] = count
    count = count + 1
print(class_dict)

X, y = [], []
for name, training_files in person_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        scalled_raw_img = cv2.resize(img, (32, 32))
        X.append(scalled_raw_img)
        y.append(class_dict[name])

X = np.array(X)
X = X/255
print(X.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# import tensorflow as tf
# import keras
from tensorflow import keras
# from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt
# import numpy as np

y = np.array(y)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

cnn = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(filters=32, kernel_size=(2, 2), activation='relu'),
    keras.layers.Conv2D(filters=32, kernel_size=(2, 2), activation='relu'),
    keras.layers.Conv2D(filters=64, kernel_size=(2, 2), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(filters=64, kernel_size=(2, 2), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),


    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5, activation='softmax')
])
cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

cnn.epochs = 40
pipe = Pipeline([('cnn', cnn)])
pipe.fit(X_train, y_train, cnn__epochs=40)

predictions = pipe.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
# y_pred
acc = np.mean(y_pred == y_test)
print("Accuracy: %.2f%%" % (acc*100))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

import joblib
# Save the model as a pickle in a file
joblib.dump(pipe, 'new_model.pkl')