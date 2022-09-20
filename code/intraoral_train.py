
"""#Preprocessing"""

import os
import matplotlib.pyplot as plt
import scipy
from datetime import datetime
import tensorflow as tf
from keras.layers.core import Dense, Flatten
from keras.layers import Dropout
from keras.models import Model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet152, ResNet101
from tensorflow.keras.applications.resnet_v2 import ResNet101V2, ResNet152V2
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications import EfficientNetB4, InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from tensorflow.compat.v1 import ConfigProto
from tensorflow.python.keras.backend import set_session

from tensorflow_addons.losses import SigmoidFocalCrossEntropy

class_names = ['frontal', 'frontal_90', 'frontal_180', 'frontal_270', 'left', 'left_90', 'left_180', 'left_270', 'lower_v2', 'lower_90_v2', 'lower_180_v2', 'lower_270_v2', 'others', 'right', 'right_90', 'right_180', 'right_270', 'upper_v2', 'upper_90_v2', 'upper_180_v2', 'upper_270_v2']
# class_names = ['frontal', 'frontal_90', 'frontal_180', 'frontal_270', 'left', 'left_90', 'left_180', 'left_270', 'lower', 'lower_90', 'lower_180', 'lower_270', 'others', 'right', 'right_90', 'right_180', 'right_270', 'upper', 'upper_90', 'upper_180', 'upper_270']
train_path = "../data/angle/train"

target_size = (224,224)
batch_size = 32 #originally 20
class_num = len(class_names)
config = ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
graph = tf.compat.v1.get_default_graph()
set_session(sess)

## size of training set divided by the class
dic = {}
for className in class_names:
    dir = train_path+"/"+className
    files = os.listdir(dir)
    dic[className] = len(files)
# plt.bar(range(len(dic)), list(dic.values()), align='center')
# plt.xticks(range(len(dic)), list(dic.keys()))
# print(dic)
# plt.show()

datagen = ImageDataGenerator(
    rescale = 1./255,
    brightness_range=[0.5,1.5],
    channel_shift_range=10,
    fill_mode = "constant",
    validation_split=0.1)

train_batches = datagen.flow_from_directory(
        train_path,
        target_size = target_size,  
        batch_size = batch_size,
        classes = class_names,
        shuffle=True,
        subset='training')

valid_batches = datagen.flow_from_directory(
        train_path,
        shuffle=False,
        target_size = target_size,
        batch_size = batch_size,
        classes = class_names,
        subset='validation')

# print(train_batches.class_indices)

# fig = plt.figure(figsize=(15, 15))
# for j in batch_size:
#     image = train_batches[0][0][j]
#     fig.add_subplot(6, 6, j + 1)
#     plt.imshow(image)
#     plt.axis('off')

###########################################################################
"""#Model"""

# 凍結網路層數
FREEZE_LAYERS = 0


net = EfficientNetB4(include_top=False,
                weights="imagenet",
                input_tensor=None,
                input_shape=(target_size[0], target_size[1], 3),
                pooling="avg",
                classes=class_num)

x = net.output
x = Flatten()(x)
x = Dropout(0.3)(x)
output_layer = Dense(class_num, activation='softmax', name='softmax')(x)

# 設定凍結與要進行訓練的網路層
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True

# print(net_final.summary())
########################################################################################
"""#Training"""

model_checkpoint = ModelCheckpoint(
    filepath='checkpoint_{epoch}.h5',
    save_weights_only=True,
    monitor='val_accuracy',
    verbose=1,
    mode='auto',
    save_freq='epoch',
    save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=1e-7, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
logger = CSVLogger("training_log_" + str(datetime.now().strftime('%H:%M')) + ".csv", append=True, separator=',')


optimizer = Adam(learning_rate=1e-3)
# optimizer = AdamW(learning_rate=1e-2, weight_decay=0.004)


net_final.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# net_final.compile(optimizer=optimizer, loss=SigmoidFocalCrossEntropy(), metrics=['accuracy'])

history = net_final.fit(train_batches,
            steps_per_epoch = train_batches.samples // batch_size,
            validation_data = valid_batches,
            validation_steps = valid_batches.samples // batch_size,
            verbose = 1,
            callbacks = [model_checkpoint, reduce_lr, logger],
            epochs = 40)

# print(history.history)
"""#Saving and Checking


"""

net_final_json = net_final.to_json()
model_name = 'intraoral_inMouth_angle'
with open("../models/"+ model_name+".json", 'w') as json_file:
    json_file.write(net_final_json)
net_final.save_weights("../models/"+ model_name+".h5")
# net_final.save_model('inMouth_angle')


STEP_SIZE_VALID = valid_batches.n // valid_batches.batch_size
result = net_final.evaluate_generator(generator=valid_batches, steps=STEP_SIZE_VALID, verbose=1)

print("result = ", result) #loss and acc

##Accuracy
fig = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='lower right')
# plt.show()
fig.savefig('accuracy_' + str(datetime.now().strftime('%H:%M')) + '.jpg')
plt.close(fig)


##Loss
fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
# plt.show()
fig.savefig('loss_'+ str(datetime.now().strftime('%H:%M')) +'.jpg')
plt.close(fig)

