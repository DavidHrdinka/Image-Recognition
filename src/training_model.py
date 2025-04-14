import tensorflow as tf
import keras
from keras import layers, regularizers, metrics, models
from keras.applications import ResNet50
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

dataset_path = "/home/aqehk/Image-Recognition/images/cardio"
image_size = (60, 80)
batch_size = 32
num_classes = 12
BUFFER_SIZE = 2500


def load_dataset(dataset_path):
    dataset = keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        image_size=image_size,
        batch_size=None,
        color_mode="grayscale",
        shuffle=True
    )
    images, labels = zip(*[(img.numpy(), label.numpy()) for img, label in dataset])
    return np.array(images), np.array(labels)


def preprocess_data(images, labels):
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, stratify=labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 / 3, stratify=y_temp, random_state=42)

    X_train = np.repeat(X_train, 3, axis=-1)
    X_val = np.repeat(X_val, 3, axis=-1)
    X_test = np.repeat(X_test, 3, axis=-1)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_dataset(X, y):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model():
    base_model = ResNet50(include_top=False, input_shape=(60, 80, 3), weights=None)

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def plot_training_history(history):
    history_dict = history.history
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history_dict['accuracy'], label='Train Accuracy')
    plt.plot(history_dict['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_dict['loss'], label='Train Loss')
    plt.plot(history_dict['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_plot.png")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(num_classes),
                yticklabels=np.arange(num_classes))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig("confusion_matrix.png")


class F1Score(metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p, r = self.precision.result(), self.recall.result()
        return 2 * (p * r) / (p + r + tf.keras.backend.epsilon())

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


images, labels = load_dataset(dataset_path)
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(images, labels)
train_ds, val_ds, test_ds = map(prepare_dataset, [X_train, X_val, X_test], [y_train, y_val, y_test])

data_augmentation = keras.Sequential([
    layers.RandomRotation(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1, 0.2),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomContrast(0.2),
])


train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

model = build_model()
model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', metrics.Precision(name="precision"), metrics.Recall(name="recall"), F1Score(name="f1_score")]
)


history = model.fit(train_ds, validation_data=val_ds, epochs=50)

test_loss, test_accuracy, test_precision, test_recall, test_f1 = model.evaluate(test_ds)
print(
    f"Test Loss: {test_loss}\nTest Accuracy: {test_accuracy}\nTest Precision: {test_precision}\nTest Recall: {test_recall}\nTest F1-Score: {test_f1}")

model.save('ultrasound-view-predictor.h5')

plot_training_history(history)

y_true, y_pred = [], []
for images, labels in test_ds:
    predictions = model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

plot_confusion_matrix(y_true, y_pred)