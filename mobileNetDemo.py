
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping

pickle_in=open("X_demo.pickle","rb")
X=pickle.load(pickle_in)
pickle_in.close()

pickle_in=open("y_demo.pickle","rb")
y=pickle.load(pickle_in)
pickle_in.close()

pickle_in=open("X_demo_test.pickle","rb")
X_test=pickle.load(pickle_in)
pickle_in.close()

pickle_in=open("y_demo_test.pickle","rb")
y_test=pickle.load(pickle_in)
pickle_in.close()

X_train,X_val,Y_train,Y_val= train_test_split(X, y, test_size=0.3)

X_final = np.stack((X_train,)*3, axis=-1)
y_final = to_categorical(Y_train, num_classes=2)
print(X_final.shape)
print(y_final.shape)

X_valid = np.stack((X_val,)*3, axis=-1)
y_valid = to_categorical(Y_val, num_classes=2)
print(X_valid.shape)
print(y_valid.shape)

X_test_final = np.stack((X_test,)*3, axis=-1)
y_test_final = to_categorical(y_test, num_classes=2)
print(X_test_final.shape)
print(y_test_final.shape)



def mobileNet():
    print("Loading mobilenet Model...")
    model = Sequential()
    model.add(MobileNet(include_top=False,weights='imagenet', pooling='avg'))
    #model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.summary()
    Training(model)
def Training(model):
    model=model
    filepath = "demo-mobilenet-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    print("Starting Training... ")
    history = model.fit(X_final, y_final,
                    epochs=500,
                    batch_size=5
                    ,validation_data=(X_valid, y_valid),
                    callbacks=callbacks_list
                       )
    Evaluate(model)

def Evaluate(model):
    print("Evalutaing Model on Test Set... ")
    model.evaluate(X_test_final, y_test_final)



mobileNet()

