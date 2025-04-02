import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt


# 数据准备函数
def load_train_data(data_dir, csv_path):
    """
    加载训练集数据，图片存放在子文件夹中
    """
    data = pd.read_csv(csv_path)

    images = []
    labels = []

    for idx, row in data.iterrows():
        # 构建完整路径
        img_path = os.path.join(data_dir, row['Path'])

        try:
            # 读取图像并预处理
            img = Image.open(img_path)
            img = img.crop((row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']))  # 裁剪ROI区域
            img = img.resize((32, 32))  # 统一尺寸
            img_array = np.array(img) / 255.0  # 归一化

            # 处理不同通道数的图像
            if len(img_array.shape) == 2:  # 灰度图转RGB
                img_array = np.stack((img_array,) * 3, axis=-1)
            elif img_array.shape[2] == 4:  # RGBA转RGB
                img_array = img_array[:, :, :3]

            images.append(img_array)
            labels.append(row['ClassId'])
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

    return np.array(images), np.array(labels)


def load_flat_data(data_dir, csv_path):
    """
    加载验证集和测试集数据，图片直接存放在文件夹下
    """
    data = pd.read_csv(csv_path)

    images = []
    labels = []

    for idx, row in data.iterrows():
        # 构建完整路径
        img_path = os.path.join(data_dir, row['Path'])

        try:
            # 读取图像并预处理
            img = Image.open(img_path)
            img = img.resize((32, 32))  # 统一尺寸
            img_array = np.array(img) / 255.0  # 归一化

            # 处理不同通道数的图像
            if len(img_array.shape) == 2:  # 灰度图转RGB
                img_array = np.stack((img_array,) * 3, axis=-1)
            elif img_array.shape[2] == 4:  # RGBA转RGB
                img_array = img_array[:, :, :3]

            images.append(img_array)
            labels.append(row['ClassId'])
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

    return np.array(images), np.array(labels)


# 数据增强
def create_datagen():
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,  # 交通标志通常不需要水平翻转
        fill_mode='nearest'
    )


# 构建改进的CNN模型
def build_model(input_shape=(32, 32, 3), num_classes=43):
    model = models.Sequential([
        # 第一卷积块
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # 第二卷积块
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # 第三卷积块
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # 分类器
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


# 训练模型
def train_model():
    # 路径设置
    base_dir = 'D:/pycharm/CNN/new-GTSRB'
    train_csv = os.path.join(base_dir, 'Train.csv')
    meta_csv = os.path.join(base_dir, 'Meta.csv')
    test_csv = os.path.join(base_dir, 'Test.csv')

    # 加载数据
    print("Loading training data...")
    X_train, y_train = load_train_data(base_dir, train_csv)

    print("Loading validation data...")
    X_val, y_val = load_flat_data(base_dir, meta_csv)

    # 数据增强
    datagen = create_datagen()
    datagen.fit(X_train)

    # 构建模型
    print("Building model...")
    model = build_model()

    # 编译模型
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    print("Training model...")
    history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                        epochs=5,
                        validation_data=(X_val, y_val),
                        verbose=1)

    # 保存模型
    model.save('traffic_sign_model.h5')
    print("Model saved to traffic_sign_model.h5")

    # 评估模型
    print("Loading test data...")
    X_test, y_test = load_flat_data(base_dir, test_csv)

    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc:.4f}')

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.savefig('training_curve.png')
    plt.show()

    return model


if __name__ == '__main__':
    train_model()