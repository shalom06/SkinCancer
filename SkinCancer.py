import os

import cv2
import numpy as np
import tensorflow
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tqdm import tqdm

print(tensorflow.executing_eagerly())
resnet_weights_path = 'resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg16_weights_path = "/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5"

# Display the dir list
print(os.listdir())


def Dataset_loader(DIR, RESIZE):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR, IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".jpg":
            img = read(PATH)
            img = cv2.resize(img, (RESIZE, RESIZE))
            IMG.append(np.array(img) / 255.)
    return IMG


benign_train = np.array(Dataset_loader('skin-cancer-malignant-vs-benign/data/train/benign', 224))
malign_train = np.array(Dataset_loader('skin-cancer-malignant-vs-benign/data/train/malignant', 224))
benign_test = np.array(Dataset_loader('skin-cancer-malignant-vs-benign/data/test/benign', 224))
malign_test = np.array(Dataset_loader('skin-cancer-malignant-vs-benign/data/test/malignant', 224))

# Create labels
benign_train_label = np.zeros(len(benign_train))
malign_train_label = np.ones(len(malign_train))
benign_test_label = np.zeros(len(benign_test))
malign_test_label = np.ones(len(malign_test))

# Merge data
X_train = np.concatenate((benign_train, malign_train), axis=0)
Y_train = np.concatenate((benign_train_label, malign_train_label), axis=0)
X_test = np.concatenate((benign_test, malign_test), axis=0)
Y_test = np.concatenate((benign_test_label, malign_test_label), axis=0)

# Shuffle train data
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

# Split validation data from train data
# x_train, x_val, y_train, y_val = train_test_split(X_train,Y_train,test_size=0.33,random_state=42)
x_train = X_train[1000:]
x_val = X_train[:1000]
y_train = Y_train[1000:]
y_val = Y_train[:1000]

# Shuffle man data
s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
Y_test = Y_test[s]

# w=60
# h=40
# fig=plt.figure(figsize=(15, 15))
# columns = 4
# rows = 3

# for i in range(1, columns*rows +1):
#     ax = fig.add_subplot(rows, columns, i)
#     if Y_train[i] == 0:
#         ax.title.set_text('Benign')
#     else:
#         ax.title.set_text('Malignant')
#     plt.imshow(x_train[i], interpolation='nearest')
# plt.show()


# datagen.fit(x_train)


# model = Sequential()
# # vgg-16 , 80% accuracy with 100 epochs
# # model.add(VGG16(input_shape=(224,224,3),pooling='avg',classes=1000,weights=vgg16_weights_path))
# # resnet-50 , 87% accuracy with 100 epochs
# model.add(ResNet50(include_top=False, input_tensor=None, input_shape=(224, 224, 3), pooling='avg', classes=2))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(BatchNormalization())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(BatchNormalization())
# model.add(Dense(1, activation='sigmoid'))
model = Sequential()

# 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

# 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
model.add(Dense(2, activation='softmax'))

# Say not to train first layer (ResNet) model as it is already trained


model.layers[0].trainable = False
model.summary()

model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

red_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.9)

batch_size = 64
epochs = 60
History = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), validation_data=(x_val, y_val),
                    epochs=epochs, steps_per_epoch=x_train.shape[0] // batch_size, verbose=1,
                    callbacks=[red_lr]
                    )

model.save('cnn.h5')
