import numpy as np
import tensorflow as tf
import classifier as cl
import utils, sys
from tensorflow import keras
from data_preprocess import get_data
import compas_data as compas


seeds = np.load('../seeds.npy')


def fit(i = 0):
    data_seed = seeds[i, 0]
    expt_seed = seeds[i, 1]
    x_train, x_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train,\
          y_race_test, feature_names = compas.get_compas_train_test(random_state = data_seed)

     

    x_train = tf.cast(x_train, dtype = tf.float32)
    y_train = y_train.astype('int32')
    y_train = tf.one_hot(y_train, 2)


    print(f'Running data seed {data_seed} and expt seed {expt_seed}')
    init_graph = utils.ClassifierGraph([50,], 2, input_shape=(9, ), seed_model = expt_seed)
    graph = cl.Classifier(init_graph, x_train,\
         y_train, num_steps = 8000, seed=expt_seed) # use for unfair algo
    graph.model.save(f'graphs/graph_{data_seed}_{expt_seed}')

if __name__ == '__main__':

      i = int(float(sys.argv[1]))
      fit(i)