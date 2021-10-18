import sys 
import os
import gc
import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

INLIER = 4
SEED = 17
INLIER = 8

MASK_MODEL_PATH = sys.argv[1]
TRAIN_IMG_PATH = sys.argv[2]
TRAIN_LABEL_PATH = sys.argv[3]
TEST_IMG_PATH = sys.argv[4]
TEST_LABEL_PATH = sys.argv[5]

TRAIN_PERC = .6667
OUTLIER_PERC = .30

os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def get_data(inlier):
    x_train = np.load(TRAIN_IMG_PATH).reshape((-1, 28, 28, 1))
    y_train = np.load(TRAIN_LABEL_PATH)
    x_test = np.load(TEST_IMG_PATH).reshape((-1, 28, 28, 1))
    y_test = np.load(TEST_LABEL_PATH)
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    x = np.concatenate((x_train, x_test)).astype("float32")
    x = x
    y = np.concatenate((y_train, y_test)).astype("float32")

    inl = x[y == inlier]
    out = x[y != inlier]

    n_inl = inl.shape[0]
    n_out = out.shape[0]

    train_inl_ind = np.random.choice(range(inl.shape[0]), int(n_inl * TRAIN_PERC), replace=False)

    train_inl = inl[train_inl_ind]
    test_inl = np.delete(inl, train_inl_ind, 0)
    n_test_inl = test_inl.shape[0]

    val_inl_ind = np.random.choice(range(train_inl.shape[0]), 300, replace=False)
    val_inl = train_inl[val_inl_ind]

    train_inl = np.delete(train_inl, val_inl_ind, 0)

    test_out_ind = np.random.choice(range(out.shape[0]), int(n_test_inl * OUTLIER_PERC), replace=False)
    test_out = out[test_out_ind]

    test_imgs = np.concatenate((test_inl, test_out)).astype("float32")
    test_inl_labels = np.zeros(test_inl.shape[0]).astype(int)
    test_out_labels = np.ones(test_out.shape[0]).astype(int)
    test_labels = np.concatenate((test_inl_labels, test_out_labels))

    return test_inl, test_out

inl_test, out_test = get_data(INLIER)

inl_test = inl_test[:500]
out_test = out_test[:500]
print(inl_test.shape, out_test.shape)

inl_sample = inl_test[:5]
out_sample = out_test[:5]
print(inl_sample.shape, out_sample.shape)

mask_model = tf.keras.models.load_model(MASK_MODEL_PATH)
mask_model.summary()

ex_inl_res = mask_model(inl_sample).numpy()

inl_pred = mask_model(inl_test).numpy()
out_pred = mask_model(out_test).numpy()
inl_pred = np.array([item.flatten() for item in list(inl_pred)]).flatten()
inl_pred = tf.math.divide(tf.math.subtract(inl_pred , inl_pred.min()), tf.math.subtract(inl_pred.max(), inl_pred.min())).numpy()
out_pred = np.array([item.flatten() for item in list(out_pred)]).flatten()
out_pred = tf.math.divide(tf.math.subtract(out_pred , out_pred.min()), tf.math.subtract(out_pred.max(), out_pred.min())).numpy()
print(inl_pred.shape, inl_pred.min(), inl_pred.max())
print(out_pred.shape, out_pred.min(), out_pred.max())

bin_inl_test = (inl_test != 0).astype(int)
bin_out_test = (out_test != 0).astype(int)

labels_inl = np.array([item.flatten() for item in list(bin_inl_test)]).flatten()
labels_out = np.array([item.flatten() for item in list(bin_out_test)]).flatten()

m = tf.keras.metrics.AUC(num_thresholds = 100)
m.update_state(labels_inl, inl_pred)
auc = m.result().numpy()
print(auc)

m2 = tf.keras.metrics.AUC(num_thresholds = 100)
m2.update_state(labels_out, out_pred)
auc2 = m2.result().numpy()
print(auc2)

print("Inlier AUC", auc)
print("Outlier AUC", auc2)
