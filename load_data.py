import os
import sys
import cv2
import csv
import random
import numpy as np

def load_data(load_type, image_dir):
    assert(load_type in ['test', 'dev', 'train'])

    x = np.empty((25,300),float)
    y = np.empty((1,300),float)

    # load word vectors
    vectors = np.load('word2vec.dat.wv.vectors.npy')
    word_list = [w.lower().strip() for w in open("kirkby_wv.txt")]
    word_to_index = {w: i for i, w in enumerate(word_list)}
    def word_to_vector(word):
        if word == "---":
            return np.zeros(vectors[0].shape)
        return vectors[word_to_index[word]]

    # load from generated_data/train_examples.csv
    csv_path = os.path.join(image_dir, load_type + "_examples.csv")
    with open(csv_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            boards_matrix = np.empty((0, 300), float)
            # add guess word vector to labels
            guess = line[0]
            
            print(y.shape)
            print(word_to_vector(guess).reshape(300,1).T.shape)
            y = np.stack((y, word_to_vector(guess).reshape(300,1).T))

            # add word vector for each word in board to input
            for word in line[1:]:
                word_vec = word_to_vector(word).T
                print(boards_matrix.shape)
                boards_matrix = np.vstack((boards_matrix, word_vec))
            print(x.shape)
            print(boards_matrix.shape)
            x = np.stack((x, boards_matrix))
        #print(x)
        #print(y)
        print(x.shape)
        print(y.shape)

            #path, label = line
            #assert(os.path.exists(path))

            #if random.random() < keep_prob:
            #    img = cv2.resize(cv2.imread(path), img_shape) / 256.
            #    if grayscale:
            #        img = np.expand_dims(img[:,:,0], 3)

            #    x.append(img)
            #    y.append(float('cancer' in label))

    #print("Data for {} set with {} examples. {}% have positive labels and {}% have negative labels.".format(load_type, x.shape[0], np.mean(y), 1 - np.mean(y)))
    return x, y
  
if __name__ == "__main__":
    load_data('train', 'generated_data')
