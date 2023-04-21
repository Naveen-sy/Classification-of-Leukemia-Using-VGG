from keras.applications import VGG16
from keras.models import Model
#from keras.layers import Dense, Flatten, Dropout
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Input
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import cv2
import os
import pickle

# Parameters
EPOCHS = 50
IMAGE_DIMS = (96, 96, 3)
BATCH_SIZE = 16
LR = 0.001

# Input images
imagePaths = sorted(list(paths.list_images("Dataset")))
np.random.seed(42)
np.random.shuffle(imagePaths)

# Create list of data and labels
data = []
labels = []

# Loop over the input images
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    data.append(image)

    label = imagePath.split(os.path.sep)[-2].split("_")
    labels.append(label)

# Convert data and labels to numpy arrays
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Binarize the labels
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# Split the data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the VGG-M model
input_tensor = Input(shape=IMAGE_DIMS)
x = Conv2D(96, (7, 7), strides=(2, 2), activation='relu', padding='same')(input_tensor)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)
x = Conv2D(256, (5, 5), strides=(2, 2), activation='relu', padding='same')(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)
x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
output_tensor = Dense(trainY.shape[1], activation='sigmoid')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)

# Compile the model
opt = Adam(lr=LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


# fitting the model 
hist = model.fit(x=trainX,y=trainY,epochs =EPOCHS ,batch_size = 128,validation_data =(testX,testY),verbose = 1)

# evaluate the model
test_score = model.evaluate(testX,testY)
print("Test loss {:.5f},accuracy {:.3f} ".format(test_score[0],test_score[1]*100))

# Save history to a pickle file
with open("history.pickle", "wb") as f:
    pickle.dump(hist.history, f)

model.save('TRAINING_EXPERIENCE.h5')

f = open("MLB.PICKLE", "wb")
f.write(pickle.dumps(mlb))
f.close()

#plt.style.use("ggplot")
#plt.figure()
plt.plot(np.arange(0, EPOCHS), hist.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), hist.history["val_loss"], label="val_loss")
plt.title("Loss Graph")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.show()

#plt.style.use("ggplot")
#plt.figure()
plt.plot(np.arange(0, EPOCHS), hist.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), hist.history["val_accuracy"], label="val_acc")
plt.title("Accuracy Graph")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.legend()
plt.show()
