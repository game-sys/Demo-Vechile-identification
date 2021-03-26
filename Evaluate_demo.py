
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np


# DATADIR="SplitedFinalData/test/"
CATEGORIES=['Toyota_corolla_2011','suzuki_alto_2007']
# IMG_SIZE=224

pickle_in=open("X_demo_test.pickle","rb")
X_test=pickle.load(pickle_in)
pickle_in.close()

pickle_in=open("y_demo_test.pickle","rb")
y_test=pickle.load(pickle_in)
pickle_in.close()

X_test_final = np.stack((X_test,)*3, axis=-1)
y_test_final = to_categorical(y_test, num_classes=2)
print(X_test_final.shape)
print(y_test_final.shape)

from tensorflow import keras
model = keras.models.load_model('demo-mobilenet-10-0.97.hdf5')

Y_test = np.argmax(y_test_final, axis=1) # Convert one-hot to index
y_pred = model.predict_classes(X_test_final)

print(X_test_final.shape)
print(classification_report(Y_test, y_pred))
print(accuracy_score(Y_test, y_pred))
#print(model.predict_classes(X_test_final))

cm = confusion_matrix(Y_test, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
acc=(cm.diagonal())
res = dict(zip(CATEGORIES, acc))
#acc=np.sort(acc)
print(res)
print(dict(sorted(res.items(), key=lambda item: item[1])))
