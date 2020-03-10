import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

np.random.seed(11)  # It's my lucky number

from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding

from keras.models import load_model

model = load_model('cancer.h5')

folder_benign_train = 'skin-cancer-malignant-vs-benign/data/train/benign'
folder_malignant_train = 'skin-cancer-malignant-vs-benign/data/train/malignant'

folder_benign_test = 'skin-cancer-malignant-vs-benign/data/test/benign'
folder_malignant_test = 'skin-cancer-malignant-vs-benign/data/test/malignant'

folder = 'skin-cancer-malignant-vs-benign/man'

read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
readC = lambda imname: np.asarray(Image.open(imname).resize((224, 224), Image.ANTIALIAS).convert("RGB"))

# im = Image.open("man.jpg")
# np_im = numpy.array(im)
# print (np_im.shape)
# ims_test='man.jpg'
# im = Image.open(ims_test)
# imResize = im.resize((224, 224), Image.ANTIALIAS)
# imgRgb=imResize.convert("RGB")


ims_outer = [readC(os.path.join(folder, filename)) for filename in os.listdir(folder)]

X_Test_Image = np.array(ims_outer, dtype='uint8')

# Load in training pictures
ims_benign = [read(os.path.join(folder_benign_train, filename)) for filename in os.listdir(folder_benign_train)]
X_benign = np.array(ims_benign, dtype='uint8')
ims_malignant = [read(os.path.join(folder_malignant_train, filename)) for filename in
                 os.listdir(folder_malignant_train)]
X_malignant = np.array(ims_malignant, dtype='uint8')

# Load in testing pictures
ims_benign = [read(os.path.join(folder_benign_test, filename)) for filename in os.listdir(folder_benign_test)]
X_benign_test = np.array(ims_benign, dtype='uint8')
ims_malignant = [read(os.path.join(folder_malignant_test, filename)) for filename in os.listdir(folder_malignant_test)]
X_malignant_test = np.array(ims_malignant, dtype='uint8')

# Create labels
y_benign = np.zeros(X_benign.shape[0])
y_malignant = np.ones(X_malignant.shape[0])

y_benign_test = np.zeros(X_benign_test.shape[0])
y_malignant_test = np.ones(X_malignant_test.shape[0])

# Merge data
X_train = np.concatenate((X_benign, X_malignant), axis=0)
y_train = np.concatenate((y_benign, y_malignant), axis=0)

X_test = np.concatenate((X_benign_test, X_malignant_test), axis=0)
y_test = np.concatenate((y_benign_test, y_malignant_test), axis=0)

# Shuffle data
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
y_train = y_train[s]

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
y_test = y_test[s]

w = 40
h = 30
fig = plt.figure(figsize=(12, 8))
columns = 5
rows = 3

# for i in range(1, columns*rows +1):
#     ax = fig.add_subplot(rows, columns, i)
#     if y_train[i] == 0:
#         ax.title.set_text('Benign')
#     else:
#         ax.title.set_text('Malignant')
#     plt.imshow(X_train[i], interpolation='nearest')
# plt.show()

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

X_train = X_train / 255.
X_test = X_test / 255.
#
#
# def build(input_shape= (224,224,3), lr = 1e-3, num_classes= 2,
#           init= 'normal', activ= 'relu', optim= 'adam'):
#     model = Sequential()
#     model.add(Conv2D(64, kernel_size=(3, 3),padding = 'Same',input_shape=input_shape,
#                      activation= activ, kernel_initializer='glorot_uniform'))
#     model.add(MaxPool2D(pool_size = (2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(64, kernel_size=(3, 3),padding = 'Same',
#                      activation =activ, kernel_initializer = 'glorot_uniform'))
#     model.add(MaxPool2D(pool_size = (2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu', kernel_initializer=init))
#     model.add(Dense(num_classes, activation='softmax'))
#     model.summary()
#
#     if optim == 'rmsprop':
#         optimizer = RMSprop(lr=lr)
#
#     else:
#         optimizer = Adam(lr=lr)
#
#     model.compile(optimizer = optimizer ,loss = "binary_crossentropy", metrics=["accuracy"])
#     return model
#
# # Set a learning rate annealer
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
#                                             patience=5,
#                                             verbose=1,
#                                             factor=0.5,
#                                             min_lr=1e-7)
#
# input_shape = (224, 224, 3)
# lr = 1e-5
# init = 'normal'
# activ = 'relu'
# optim = 'adam'
# epochs = 50
# batch_size = 64
#
# model = build(lr=lr, init=init, activ=activ, optim=optim, input_shape=input_shape)
#
# history = model.fit(X_train, y_train, validation_split=0.2,
# #                     epochs=epochs, batch_size=batch_size, verbose=1,
# #                     callbacks=[learning_rate_reduction]
# #                     )
#
# model.save('cancer.h5')
# # # list all data in history
# print(model.history.keys())
# # summarize history for accuracy
# plt.plot(model.history['acc'])
# plt.plot(model.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'man'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(model.history['loss'])
# plt.plot(model.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'man'], loc='upper left')
# plt.show()

print(model.predict(X_Test_Image))
