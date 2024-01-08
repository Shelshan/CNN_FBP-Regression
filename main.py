import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetV2B3, EfficientNetV2L, ConvNeXtBase, ConvNeXtLarge
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import os
import sys
import tensorflow as tf
import pandas as pd
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def make_print_to_file(path='./'):
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            pass
    fileName = "SCUT-FBP_ConvNeXtBase"
    # fileName = "SCUT-FBP5500_ConvNeXtLarge"
    sys.stdout = Logger(fileName + '.log', path=path)
    print(fileName.center(60,'*'))

# loss function
def smooth_mse(y_true, y_pred):
    """Smooth Mean Squared Error"""
    smooth_factor = 0.1
    mse = K.mean(K.square(y_true - y_pred))
    abs_diff = K.abs(y_true - y_pred)
    smooth_mse = mse + smooth_factor * K.mean(abs_diff)
    return smooth_mse

# metrics
def pearson_correlation(x, y):
    x_mean = K.mean(x)
    y_mean = K.mean(y)
    x_std = K.std(x)
    y_std = K.std(y)
    covariance = K.mean((x - x_mean) * (y - y_mean))
    correlation = covariance / (x_std * y_std + K.epsilon())  # 添加一个小的常数以防除以零
    return correlation

def rmse(y_true, y_pred):
    loss = K.sqrt(K.mean((y_pred - y_true) ** 2))
    return loss

def mae(y_true, y_pred): 
    loss = K.mean(K.abs(y_true - y_pred))
    return loss

if __name__ == '__main__':

    make_print_to_file("./results")
    
    # get_data
    data = pd.read_csv('./datasets/SCUT-FBP5500/All_labels.txt', sep=" ", header=None)
    data_np = data.values

    imgs = np.empty((len(data_np), 224, 224, 3), dtype=np.uint8)
    for i, (img_filename, rating) in enumerate(data_np):
        img = cv2.imread(os.path.join('./datasets/SCUT-FBP5500/Images', img_filename))
        img = cv2.resize(img, (224, 224))
        imgs[i] = img

    x_train, x_val, y_train, y_val = train_test_split(imgs, data_np[:, 1], test_size=0.2)
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

    # data_augs
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,  # 随机旋转角度范围
        width_shift_range=0.2,  # 随机水平平移范围
        height_shift_range=0.2,  # 随机垂直平移范围
        shear_range=0.2,  # 随机剪切强度
        zoom_range=0.2,  # 随机缩放范围
        horizontal_flip=True  # 水平翻转
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255)

    train_generator = train_datagen.flow(
        x=x_train, y=y_train,
        batch_size=16,
        shuffle=True)

    val_generator = val_datagen.flow(
        x=x_val, y=y_val,
        batch_size=16,
        shuffle=False)

    # 构建模型
    base_model = EfficientNetV2B3(include_top=False,
                                    weights="imagenet",
                                    input_tensor=None,
                                    input_shape=(224, 224, 3),
                                    pooling='avg',
                                    include_preprocessing=True)

    # base_model = EfficientNetV2L(include_top=False,
    #                             weights="imagenet",
    #                             input_tensor=None,
    #                             input_shape=(224, 224, 3),
    #                             pooling='avg',
    #                             include_preprocessing=True)

    # base_model = ConvNeXtBase(model_name="convnext_base",
    #                         include_top=False,
    #                         include_preprocessing=True,
    #                         weights="imagenet",
    #                         input_tensor=None,
    #                         input_shape=(224, 224, 3),
    #                         pooling='avg')

    # base_model = ConvNeXtLarge(model_name="convnext_large",
    #                             include_top=False,
    #                             include_preprocessing=True,
    #                             weights="imagenet",
    #                             input_tensor=None,
    #                             input_shape=(224, 224, 3),
    #                             pooling='avg')

    # base_model.layers[0].trainable = False
    model = models.Sequential([base_model,
                                layers.Dense(256, activation = 'relu'),
                                layers.Dense(64, activation = 'relu'),
                                layers.Dense(1, activation='relu')])

    # 编译模型
    # model.compile(optimizer=Adam(learning_rate = 1e-4), loss = smooth_mse, metrics = [pearson_correlation, mae, rmse])
    model.compile(optimizer=Adam(learning_rate = 1e-5), loss = smooth_mse, metrics = [pearson_correlation, mae, rmse])
    model.summary()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,verbose=1,
                                min_delta=1e-4,min_lr=1e-8,mode='min')
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=30,
                                                    mode='min', verbose=1)
    callbacks = [earlystopping, reduce_lr]

    # 训练模型
    history = model.fit(train_generator, 
                        steps_per_epoch=len(train_generator), 
                        epochs=50, 
                        validation_data=val_generator, 
                        callbacks=callbacks,
                        shuffle=False)