import tensorflow as tf
from imutils import paths
import numpy as np
import json
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa

np.random.seed(100)
unique_labels = None


def get_train_test_val_set(data, data_labels, test_size=0.25):
    # Split train/val/test dataset
    X_train, X_test, y_train, y_test = train_test_split(data, data_labels, test_size=test_size, stratify=data_labels,
                                                        random_state=24,
                                                        shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=23,
                                                    shuffle=True)
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_pair(images_path, images_labels):
    code_book = {}
    [code_book.update({name: [i]})
     if name not in code_book.keys()
     else code_book[name].append(i)
     for i, name in enumerate(images_labels)]

    uniform_pairs = []
    first_half = []
    second_half = []

    for index, current_image in enumerate(images_labels):
        same_img_index = np.random.choice(code_book[current_image])
        uniform_pairs.append((images_path[index], images_path[same_img_index]))

        diff_img = np.random.choice(list(code_book.keys()))
        # choosing different image category
        while diff_img == current_image:
            diff_img = np.random.choice(list(code_book.keys()))
        diff_img_index = np.random.choice(code_book[diff_img])

        uniform_pairs.append((images_path[index], images_path[diff_img_index]))

    np.random.shuffle(uniform_pairs)
    [(first_half.append(f), second_half.append(s)) for f, s in uniform_pairs]
    return first_half, second_half


def preprocessing_image(img_path):
    global unique_labels
    img_name = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_name, channels=3)

    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (224, 224))

    label = tf.strings.split(img_path, os.path.sep)[-2]
    onehot = label == unique_labels
    label = tf.argmax(onehot)
    return img, label


def process_pair(first, second):
    first, labelF = preprocessing_image(first)
    second, labelSec = preprocessing_image(second)
    one = tf.constant(1, dtype=tf.float32)
    zero = tf.constant(0, dtype=tf.float32)
    return (first, second), one if tf.math.equal(labelF, labelSec) else zero


#
# def augment_using_ops(images, label):
#     def augment(image):
#         image = tf.image.random_flip_left_right(image)
#         image = tf.image.random_flip_up_down(image)
#         image = tf.image.rot90(image)
#         image = tf.image.random_brightness(image, 0.2)
#         image = tf.image.random_contrast(image, 0.2, 0.5)
#         return image
#
#     first_aug_img = augment(images[0])
#     second_aug_img = augment(images[1])
#
#     return (first_aug_img, second_aug_img), label


def create_data_pipeline(first_half, second_half):
    line_one = tf.data.Dataset.from_tensor_slices(first_half)
    line_two = tf.data.Dataset.from_tensor_slices(second_half)

    dataPipeline = tf.data.Dataset.zip((line_one, line_two))
    dataPipeline = (dataPipeline
                    .map(process_pair, num_parallel_calls=tf.data.AUTOTUNE)
                    .cache()
                    .batch(32)
                    .prefetch(tf.data.AUTOTUNE))
    return dataPipeline


class euclidian_dist_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(euclidian_dist_layer, self).__init__()

    def call(self, inputs):
        featsA, featsB = inputs
        sum_square = tf.math.reduce_sum(tf.math.square(featsA - featsB), axis=1, keepdims=True)
        return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


def create_embedded_model():
    vgg19 = tf.keras.applications.vgg19.VGG19(weights="imagenet", input_shape=(224, 224) + (3,), include_top=False)
    for layer in vgg19.layers:
        layer.trainable = False

    flatten = tf.keras.layers.Flatten()(vgg19.output)
    FC1 = tf.keras.layers.Dense(512, activation="relu")(flatten)
    BN1 = tf.keras.layers.BatchNormalization()(FC1)
    FC2 = tf.keras.layers.Dense(256, activation="relu")(BN1)
    BN2 = tf.keras.layers.BatchNormalization()(FC2)
    output_embedd = tf.keras.layers.Dense(180, activation="relu")(BN2)

    embeddingLayer = tf.keras.Model(inputs=vgg19.input, outputs=output_embedd)
    return embeddingLayer


def createLayerInputs():
    line_one_input = tf.keras.layers.Input(shape=(224, 224) + (3,))
    line_two_input = tf.keras.layers.Input(shape=(224, 224) + (3,))
    return line_one_input, line_two_input


def create_siamese_network():
    embeddingLayer = create_embedded_model()

    line_one_input, line_two_input = createLayerInputs()

    euclidean_dist = euclidian_dist_layer()

    distance = euclidean_dist(
        [embeddingLayer(tf.keras.applications.vgg19.preprocess_input(line_one_input)),
         embeddingLayer(tf.keras.applications.vgg19.preprocess_input(line_two_input))])

    output = tf.keras.layers.Dense(1, activation="sigmoid")(distance)
    siamese_model = tf.keras.Model(inputs=[line_one_input, line_two_input], outputs=output)
    return siamese_model


def test(model, images_path, images_labels):
    correct = 0
    subclass_image_sample = {}

    [subclass_image_sample.update({name: i})
     for i, name in enumerate(images_labels)
     if name not in subclass_image_sample.keys()]

    for index, current_image in enumerate(images_labels):
        first_half = []
        second_half = []

        for sample_image in subclass_image_sample.values():
            first_half.append(images_path[index])
            second_half.append(images_path[sample_image])
        second_half = sorted(second_half, key=lambda x: x.split('/')[-2])

        X_test = create_data_pipeline(first_half, second_half)
        result = model.predict(X_test)
        predict = unique_labels[tf.argmax(result)]
        if predict == current_image:
            correct += 1
    return correct / len(images_labels)


def plot_training(H, plotPath):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)


def main():
    global unique_labels

    args = argparse.ArgumentParser()
    args.add_argument("-d", "--dataset", dest="DATA", required=True, help="path to dataset")
    args = vars(args.parse_args())

    img_paths = list(paths.list_images(args["DATA"]))
    labels = [p.split('/')[-2] for p in img_paths]
    unique_labels = np.sort(np.unique(labels), axis=None)

    X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_val_set(img_paths, labels)

    pair_one, pair_two = make_pair(X_train, y_train)
    pair_one_val, pair_two_val = make_pair(X_val, y_val)
    X_train = create_data_pipeline(pair_one, pair_two)
    X_val = create_data_pipeline(pair_one_val, pair_two_val)

    model = create_siamese_network()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics="accuracy")

    checkpoint_filepath = '/home/thanhle/Downloads/feature_engineering_class_project/checkpoints/contrasitive_Model' \
                          '.hdf5'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    history = model.fit(X_train, validation_data=X_val, epochs=100, callbacks=[model_checkpoint_callback])
    plot_training(history, "/home/thanhle/Downloads/feature_engineering_class_project/checkpoints/plot.png")

    model.load_weights(checkpoint_filepath)
    print(test(model, X_test, y_test))

    # with open('label_encoding.json', 'w') as jwrt:
    #     json.dump(dict([(i, value) for i, value in enumerate(unique_labels)]), jwrt)


if __name__ == "__main__":
    main()
