from pathlib import Path
import random
import os


def generate_splits(directory, extension='jpg'):
    images = Path(directory).glob(f'*.{extension}')

    # TRAIN SPLIT
    train_path = "train_set"
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    # TEST SPLIT
    test_path = "test_set"
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # VAL SPLIT
    val_path = "val_set"
    if not os.path.exists(val_path):
        os.makedirs(val_path)

    images = list(images)
    print(len(images))

    for image in random.sample(images, 150):
        print(image)
        images.remove(image)
        new_path = os.path.join(train_path, os.path.basename(image))
        os.rename(image, new_path)

    for image in random.sample(list(images), 30):
        images.remove(image)
        new_path = os.path.join(test_path, os.path.basename(image))
        os.rename(image, new_path)

    for image in random.sample(list(images), 20):
        images.remove(image)
        new_path = os.path.join(val_path, os.path.basename(image))
        os.rename(image, new_path)

    print("All done!")


def count_files(directory):
    images_png = Path(directory).glob(f'*.{'png'}')
    images_jpg = Path(directory).glob(f'*.{'jpg'}')
    images = list(images_png) + list(images_jpg)
    print("There are", len(images), "in", directory)

def check_counts():
    count_files('train_set')
    count_files('test_set')
    count_files('val_set')

'''
DO NOT RUN TWICE
'''
#generate_splits('real_hands')
#generate_splits('ai_hands', extension='png')

check_counts()