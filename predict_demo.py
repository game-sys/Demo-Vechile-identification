
import cv2
import numpy as np
from tensorflow import keras

IMG_SIZE=224
CATEGORIES=['Toyota_corolla_2011','suzuki_alto_2007']
path="test1.jpg" #test image path
model = keras.models.load_model('demo-mobilenet-10-0.97.hdf5')
CATEGORIES=['Toyota_corolla_2011','suzuki_alto_2007']
img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
new_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
X_test_final = np.stack((new_img,)*3, axis=-1)
X_test_final = X_test_final.reshape(1, 224, 224, -1)
predictions = model.predict(X_test_final)
classes = np.argmax(predictions, axis = 1)
print(CATEGORIES[int(classes)])

