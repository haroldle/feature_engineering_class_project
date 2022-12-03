# IMPORTING LIBRARIES
import tensorflow as tf
from imutils import paths
import numpy as np
import json
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa

# SETTING RANDOM SEED FOR PREPRODUCIBLE WORK
np.random.seed(100)
unique_labels = None

# RETURN TRAIN/VAL/TEST DATASET GIVEN THE PATH TO DATA
def get_train_test_val_set(data, data_labels, test_size=0.25):
    # Split train/val/test dataset
    X_train, X_test, y_train, y_test = train_test_split(data, data_labels, test_size=test_size, stratify=data_labels,
                                                        random_state=24,
                                                        shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=23,
                                                    shuffle=True)
    return X_train, X_val, X_test, y_train, y_val, y_test

# CREATING PAIR OF POS/NEG IMAGES FOR THE SIAMESE MODEL TO EXTRACT THE FEATURES AND CALCULATE THE DIFFERENT BETWEEN IMAGE
# POS WHERE IMAGES ARE SIMILAR (MEANING TWO IMAGES ARE IN SAME CATEGORY)
# NEG WHERE IMAGES ARE NOT SIMILAR (MEANING TWO IMAGES ARE NOT IN SAME CATEGORY)
def make_pair(images_path, images_labels):
    # TAKING 1 SAMPLE OF EACH CATEGORY
    # CODE_BOOK JUST A BOOK CONTAINING 1 SAMPLE OF EACH CATEGORY
    code_book = {}
    [code_book.update({name: [i]})
     if name not in code_book.keys()
     else code_book[name].append(i)
     for i, name in enumerate(images_labels)]
    
    # CREATING TEMPORARY STORAGE
    uniform_pairs = []
    # IMAGINE MAKE PAIR IS MAKING TWO LINES OF PARALLEL IMAGES (FIRST HALF MEANS FIRST LINE, SECOND HALF MEANS SECOND LINE) 
    first_half = []
    second_half = []
    # LOOPING THROUGH IMAGES IN DATASET
    for index, current_image in enumerate(images_labels):
        # PICKING 1 IMAGE THAT IS IN THE SAME CATEGORY
        same_img_index = np.random.choice(code_book[current_image])
	# ADDING THE PAIR TO THE GENERAL POOL OF IMAGES
        uniform_pairs.append((images_path[index], images_path[same_img_index]))
        
        # choosing different image category
        diff_img = np.random.choice(list(code_book.keys()))
        while diff_img == current_image:
            diff_img = np.random.choice(list(code_book.keys()))
        diff_img_index = np.random.choice(code_book[diff_img])
	# ADDING THE PAIR TO THE GENERAL POOL OF IMAGES
        uniform_pairs.append((images_path[index], images_path[diff_img_index]))
    # SHUFFLING ALL PAIRS FOR RANDOMIZATION
    np.random.shuffle(uniform_pairs)
    # SPLITTING THE PAIRS INTO TWO SEPARATE LINES
    [(first_half.append(f), second_half.append(s)) for f, s in uniform_pairs]
    return first_half, second_half

# PREPROCESSED IMAGE BEFORE FEEDING TO THE MODEL
def preprocessing_image(img_path):
    global unique_labels
    # READING IMAGE FROM FILE
    img_name = tf.io.read_file(img_path)
    # DECODING JPEG IMAGE FILE
    img = tf.image.decode_jpeg(img_name, channels=3)
    # CONVERTING IMAGE DATA TYPE AND RESCALING THE IMAGE SIZE
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (224, 224))
    # GETTING THE IMAGE LABELS
    label = tf.strings.split(img_path, os.path.sep)[-2]
    onehot = label == unique_labels
    # CONVERTING THE LABELS TO ONE HOT ENCODERS
    label = tf.argmax(onehot)
    return img, label

# CREATE PAIR OF IMAGES BASED ON THE IMAGE LINES
def process_pair(first, second):
    # PROCESS FIRST LINE OF IMAGE
    first, labelF = preprocessing_image(first)
    # PROCESS SECOND LINE OF IMAGE
    second, labelSec = preprocessing_image(second)
    # TENSORFLOW ENCODE 1 AND 0 FOR LABEL (1 IS SIMILAR LABEL, 0 IS DISSIMILAR LABEL)
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

# CREATE TENSORFLOW DATA PIPELINE FOR FASTER DATA FEED SO THE MODEL CAN GET DATA FASTER => TRAIN FASTER
def create_data_pipeline(first_half, second_half):
    # CREATE IMAGE LINES FOR SIAMESE NETWORK
    line_one = tf.data.Dataset.from_tensor_slices(first_half)
    line_two = tf.data.Dataset.from_tensor_slices(second_half)
    # CREATE DATA PIPELINE FROM TWO IMAGE LINES
    dataPipeline = tf.data.Dataset.zip((line_one, line_two))
    dataPipeline = (dataPipeline
                    .map(process_pair, num_parallel_calls=tf.data.AUTOTUNE)
                    .cache()
                    .batch(32)
                    .prefetch(tf.data.AUTOTUNE))
    return dataPipeline

# SUBCLASSING EUCLIDIAN DISTANCE LAYERS FOR CALCULATING THE SIMILARITY
class euclidian_dist_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(euclidian_dist_layer, self).__init__()

    def call(self, inputs):
        featsA, featsB = inputs
        sum_square = tf.math.reduce_sum(tf.math.square(featsA - featsB), axis=1, keepdims=True)
        return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

# SUBCLASSING COSINE DISTANCE LAYERS FOR CALCULATING THE SIMILARITY
class cosine_dist_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(cosine_dist_layer, self).__init__()

    def call(self, inputs):
        featsA, featsB = inputs
        A = tf.math.l2_normalize(featsA, axis=1)
        B = tf.math.l2_normalize(featsB, axis=1)
        return tf.math.reduce_sum(tf.math.multiply(A, B), keepdims=True, axis=1)
# CREATING EMBEDDING MODEL (EMBEDDING MODEL ARE JUST PRETRAIN VGG19 WHERE IT IS GOING TO EXTRACT THE IMAGES FEATURE FROM IMAGE)
def create_embedded_model():
    # GETTING PRETRAINED VGG19
    vgg19 = tf.keras.applications.vgg19.VGG19(weights="imagenet", input_shape=(224, 224) + (3,), include_top=False)
    # SET THE LAST 3 LAYERS TO TRAINABLE FOR FINE TUNING
    vgg19.trainable = True
    for layer in vgg19.layers[:-3]:
        layer.trainable = False
    # CREATE 3 FC LAYERS FOR LEARNING THE FEATURE MAPS
    flatten = tf.keras.layers.Flatten()(vgg19.output)
    FC1 = tf.keras.layers.Dense(512, activation="relu")(flatten)
    BN1 = tf.keras.layers.BatchNormalization()(FC1)
    FC2 = tf.keras.layers.Dense(256, activation="relu")(BN1)
    BN2 = tf.keras.layers.BatchNormalization()(FC2)
    output_embedd = tf.keras.layers.Dense(128, activation="relu")(BN2)
    # CREATING THE EMBEDDING MODEL
    embeddingLayer = tf.keras.Model(inputs=vgg19.input, outputs=output_embedd)
    return embeddingLayer

# CREATING LAYER INPUT FOR SIAMESE NET
def createLayerInputs():
    line_one_input = tf.keras.layers.Input(shape=(224, 224) + (3,))
    line_two_input = tf.keras.layers.Input(shape=(224, 224) + (3,))
    return line_one_input, line_two_input

# CREATING SIAMESE NETWORK MODEL
def create_siamese_network():
    # GETTING THE EMBEDDING LAYER OR FEATURE MAP EXTRACTION
    embeddingLayer = create_embedded_model()
    # GETTING THE INPUT LAYERS FOR SIAMESE NETWORK
    line_one_input, line_two_input = createLayerInputs()
    # GETTING THE SIMILARITY DISTANCE (WHICH IS A FEATURE)
    # euclidean_dist = euclidian_dist_layer()
    cosine_dist = cosine_dist_layer()
    # distance = euclidean_dist(
    #     [embeddingLayer(tf.keras.applications.vgg19.preprocess_input(line_one_input)),
    #      embeddingLayer(tf.keras.applications.vgg19.preprocess_input(line_two_input))])

    distance = cosine_dist(
        [embeddingLayer(tf.keras.applications.vgg19.preprocess_input(line_one_input)),
         embeddingLayer(tf.keras.applications.vgg19.preprocess_input(line_two_input))])
    # CREATING AN OUTPUT TO DETERMINE THE SIMILARITY
    output = tf.keras.layers.Dense(1, activation="sigmoid")(distance)
    siamese_model = tf.keras.Model(inputs=[line_one_input, line_two_input], outputs=output)
    return siamese_model

# TESTING SIAMESE MODEL USING TEST IMAGE PATH
def test(model, images_path, images_labels):
    # NUMBER OF CORRECT LABELING IMAGE
    correct = 0
    # A CODE BOOK THAT CONTAINS 1 SAMPLE OF EACH CATEGORY
    subclass_image_sample = {}

    [subclass_image_sample.update({name: i})
     for i, name in enumerate(images_labels)
     if name not in subclass_image_sample.keys()]
    # CREATING THE TEST IMAGE PAIRS
    # FOR EACH TEST IMAGE, IT WILL PAIR WITH 46 CATEGORIES TO COMPUTE THE SIMILARITY
    # THE PAIR THAT HAS THE LOWEST DISTANCE SCORE ARE IN SAME CATEGORIES (CLASSIFICATION). THEN COMPARE THAT RESULTS TO THE ACTUAL LABELS
    for index, current_image in enumerate(images_labels):
        first_half = []
        second_half = []
	#APPENDING THE TEST IMAGE TO 46 CATEGORIES SAMPLE
        for sample_image in subclass_image_sample.values():
            first_half.append(images_path[index])
            second_half.append(images_path[sample_image])
        second_half = sorted(second_half, key=lambda x: x.split('/')[-2])
	# CREATING THE TEST DATA PIPELINE FOR MODEL TO PREDICT
        X_test = create_data_pipeline(first_half, second_half)
        # GET THE PREDICTION
        result = model.predict(X_test)
        predict = unique_labels[tf.argmax(result)]
        # COUNT THE CORRECT SAMPLES
        if predict == current_image:
            correct += 1
    # RETURN THE ACCURACY WHERE(CORRECT / TOTAL TEST SAMPLES)
    return correct / len(images_labels)


def plot_training(H, plotPath):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    # plt.plot(H.history["loss"], label="train_loss")
    # plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)


def main():
    global unique_labels
    # SET THE ARGUMENT PARSER
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--dataset", dest="DATA", required=True, help="path to dataset")
    args = vars(args.parse_args())
    # GRAB THE DATASET PATH FROM THE ARGUMENT PARSER
    img_paths = list(paths.list_images(args["DATA"]))
    # GRAB THE LABELS
    labels = [p.split('/')[-2] for p in img_paths]
    # GRAB THE UNIQUE LABELS
    unique_labels = np.sort(np.unique(labels), axis=None)
    # GRAB TRAIN/VAL/TEST SET
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_val_set(img_paths, labels)
    # GRAB IMAGE PAIRS TRAINING, VALIDATING
    pair_one, pair_two = make_pair(X_train, y_train)
    pair_one_val, pair_two_val = make_pair(X_val, y_val)
    # GRAB DATA PIPELINE FOR MODEL TO TRAIN IN TRAINING AND VALIDATING
    X_train = create_data_pipeline(pair_one, pair_two)
    X_val = create_data_pipeline(pair_one_val, pair_two_val)
    # GRAB SIAMESE MODEL
    model = create_siamese_network()
    # SET LOSS, OPTIMIZER AND METRICS
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics="accuracy")
    # SET CHECKPOINT SAVING DIRECTORY
    checkpoint_filepath = '/home/thanhle/Downloads/feature_engineering_class_project/checkpoints/Curious_model_3' \
                          '.hdf5'
    # CREATE CHECKPOINT FUNCTION FOR SAVING CHECKPOINT
    # SAVE BY BEST VALIDATION ACCURACY
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    # GETTING MODEL SUMMARY
    model.summary()
    # TRAIN MODEL
    history = model.fit(X_train, validation_data=X_val, epochs=40, callbacks=[model_checkpoint_callback])
    plot_training(history, "/home/thanhle/Downloads/feature_engineering_class_project/checkpoints/plot_3.png")
    # LOAD THE TRAINED MODEL WEIGHTS
    model.load_weights(checkpoint_filepath)
    # GET THE ACCURACY ON TEST SET
    print(test(model, X_test, y_test))

    # with open('label_encoding.json', 'w') as jwrt:
    #     json.dump(dict([(i, value) for i, value in enumerate(unique_labels)]), jwrt)


if __name__ == "__main__":
    main()
