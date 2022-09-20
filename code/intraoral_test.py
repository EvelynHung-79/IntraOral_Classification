import os
from pydoc import classname
import fnmatch
import random
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.compat.v1 import InteractiveSession, ConfigProto
from tensorflow.python.keras.backend import set_session

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
set_session(session)

path = "../data/angle/test"
class_names = ['frontal', 'frontal_90', 'frontal_180', 'frontal_270', 'left', 'left_90', 'left_180', 'left_270', 'lower', 'lower_90', 'lower_180', 'lower_270', 'others', 'right', 'right_90', 'right_180', 'right_270', 'upper', 'upper_90', 'upper_180', 'upper_270']

###########################################################

images = []
filenames = []
labels=[]
for i in range(len(class_names)):
    dir = path+"/" + class_names[i]
    for filename in sorted(os.listdir(dir)):
        img = cv2.imread(os.path.join(dir, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
        if img is not None:
            images.append(img)
            filenames.append(filename)
            labels.append(i)
    print(str(i) + ' finished')

print('finish collect images with cv2')

# fig = plt.figure(figsize=(15, 15))
# for i in range(len(images)):
#     fig.add_subplot(10, 10, i+1)
#     plt.imshow(images[i])
#     plt.axis('off')
#     plt.title(filenames[i])
#     if i == 99: break
# plt.show()
# fig.savefig('cv2_method.png')
# plt.clf()

images = np.array(images)
labels = np.array(labels)
np.save('../data/angle/test_images.npy', images)
np.save('../data/angle/test_labels.npy', labels)
images = np.load('../data/angle/test_images.npy')
labels = np.load('../data/angle/test_labels.npy')
print('Python files loaded')

###########################################################

dic = {}
for className in class_names:
    dir = path+"/"+className
    files = os.listdir(dir)
    dic[className] = len(files)

# plt.bar(range(len(dic)), list(dic.values()), align='center')
# plt.xticks(range(len(dic)), list(dic.keys()))
print(dic)
# plt.show()

target_size = (224,224)
batch_size = 1

#ImageDataGenerator() 可以做一些影像處理的動作 
datagen = ImageDataGenerator(rescale = 1./255,
                            rotation_range=0,
                            horizontal_flip=False,
                            vertical_flip=False)

#以 batch 的方式讀取資料
predict_batches = datagen.flow_from_directory(
        path,
        seed=0,
        shuffle=False,
        target_size = target_size,  
        batch_size = batch_size,
        classes = class_names)
########## ABOVE method is not working

# predict_batches = datagen.flow(
#     x=images,
#     y=labels,
#     batch_size=batch_size,
#     shuffle=False,
#     seed=0
# )

#######################################################################################
#GO Thhhhhhhhhhhhhhhhhhhrough predict_bacth print it out (lower)

# print('imageDataGenerator_method...predict_batch')
# fig = plt.figure(figsize=(15, 15))
# count = 0
# for i in range(len(predict_batches)):
#     # a = np.argmax(predict_batches[i][1])
#     b = (predict_batches.classes)[i]
#     if b == 19:
        
#         if count < 100: count += 1
#         else: break
#         image = predict_batches[i][0][0] # i th image, image(3 layers), break array
#         fig.add_subplot(10, 10, count)
#         plt.imshow(image)
#         plt.axis('off')
#     if b == 20: 
#         print('Number of images iterated', count)
#         break
# plt.show()
# fig.savefig('upper_180.png')
    
#######################################################################################


#######################################################################################
#GO Thhhhhhhhhhhhhhhhhhhrough bacth print it out (lower)
# print('imageDataGenerator_method...batch')
# fig = plt.figure(figsize=(15, 15))
# count = 0
# for i in range(len(predict_batches)):
#     b = predict_batches[i][1]
#     # b = (batch.classes)[i]
#     if b == 19:
#         if count < 100: count += 1
#         else: break
#         image = predict_batches[i][0][0] # i th image, image(3 layers), break array
#         fig.add_subplot(10, 10, count)
#         plt.imshow(image)
#         plt.axis('off')
#     elif b == 20: 
#         print('Number of images iterated', count)
#         break
#     else: continue
# plt.show()
# fig.savefig('imageDataGenerator_method_batch.png')
    
#######################################################################################


# fig = plt.figure(figsize=(15, 15))
# for i in range(len(predict_batches)):
#     for j in range(1,181):

# with open('../models/inMouth_angle.json', 'r') as json_file:
#     loaded_json = json_file.read()
# net = model_from_json(loaded_json)
# net.load_weights("../models/inMouth_angle.h5")
# print('Model is loaded.')

###########################################################################
## Iterate Through the checkpoints

def myfunc(name):
    pos1 = name.find('_') + 1
    pos2 = name.find('.')
    return int(name[pos1:pos2])

checkpoints = []

for filename in os.listdir('.'):
    if fnmatch.fnmatch(filename, 'checkpoint_*.h5'):
        checkpoints.append(filename)
checkpoints.sort(key=myfunc)
# leave only 4 checkpoints
# for i in range(len(checkpoints) - 4):
#     checkpoints.pop(0)

checkpoints.append('../models/intraoral_inMouth_angle.h5')
print(checkpoints)

checkpoint_performance = []
checkpoint_name = []

for checkpoint in checkpoints:
    print("Testing on " + checkpoint)
    with open("../models/intraoral_inMouth_angle.json", 'r') as json_file:
        loaded_json = json_file.read()
    net = model_from_json(loaded_json)
    net.load_weights(checkpoint)

    # filenames = predict_batches.filenames
    nb_samples = len(predict_batches)

    # Accuracy and Classification report
    predict = net.predict(predict_batches, steps = nb_samples, verbose = 1)
    y_pred = np.argmax(predict, axis=1)

    #Print wrong
    if checkpoint == "../models/intraoral_inMouth_angle.h5":
        print('Collecting wrong images...')
        fig = plt.figure(figsize=(15, 15))
        count = 0
        for i in range(len(predict_batches)):
            if ((predict_batches.y)[i] != y_pred[i]):
                count += 1
                image = predict_batches[i][0][0] # i th image, image(3 layers), break array
                fig.add_subplot(8, 5, count)
                plt.imshow(image)
                plt.axis('off')
                plt.title(str((predict_batches.y)[i]) + ' -> ' + str(y_pred[i]))
        plt.show()
        fig.savefig('test_wrong.png')
        plt.clf()
    

    print("Accuracy")
    accuracy = accuracy_score(predict_batches.y, y_pred)
    print(accuracy)

    checkpoint_name.append(checkpoint)
    checkpoint_performance.append(accuracy)

    print("Confusion Matrix")
    con_matrix = confusion_matrix(predict_batches.y, y_pred)
    df_cfm = pd.DataFrame(con_matrix, index = class_names, columns = class_names)
    plt.figure(figsize = (10,7))
    cfm_plot = sn.heatmap(df_cfm, annot=True)
    if checkpoint == "../models/intraoral_inMouth_angle.h5":
        cfm_plot.figure.savefig(f"confusion_matrix_final.png")
    else:
        cfm_plot.figure.savefig(f"confusion_matrix_{checkpoint[11:-3]}.png")
    

    print('\nClassification Report')
    report = classification_report(predict_batches.y, y_pred, target_names=class_names, output_dict=True)
    print(classification_report(predict_batches.y, y_pred, target_names=class_names))
    df = pd.DataFrame(report).transpose()
    if checkpoint == "../models/intraoral_inMouth_angle.h5":
        df.to_csv(f'classification_report_final.csv')
    else:
        df.to_csv(f'classification_report_{checkpoint[11:-3]}.csv')

checkpoint_name = pd.DataFrame(checkpoint_name)
checkpoint_performance = pd.DataFrame(checkpoint_performance)
df_summary = pd.concat([checkpoint_name, checkpoint_performance], axis=1)
df_summary.columns = ['Path', 'Accuracy']
df_summary.to_csv('Test_summary.csv')
print('Test Summary exported')