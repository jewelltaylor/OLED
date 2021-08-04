import tensorflow as tf
import numpy as np
import random
import os


def get_data(INLIER, BATCH_SIZE):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    print(x_train.shape)
    print(y_train.shape)

    inl = x_train[y_train.squeeze() == INLIER]
    out = x_train[y_train.squeeze() != INLIER]

    print(inl.shape)

    val_inl_ind = np.random.choice(range(inl.shape[0]), 300, replace=False)
    val_out_ind = np.random.choice(range(out.shape[0]), 300, replace=False)

    val_inl = inl[val_inl_ind]

    val_out = out[val_out_ind]

    train_inl = np.delete(inl, val_inl_ind, 0)

    train_inl = train_inl / 127.5 - 1

    print("Train Shape:", train_inl.shape)

    train_ds = tf.data.Dataset.from_tensor_slices((train_inl.astype("float32")))
    train_ds = train_ds.batch(BATCH_SIZE)

    val_inl_sample_ind = np.random.choice(range(val_inl.shape[0]), 50, replace=False)
    val_out_sample_ind = np.random.choice(range(val_out.shape[0]), 50, replace=False)
    val_inl_sample = val_inl[val_inl_sample_ind]
    val_out_sample = val_out[val_out_sample_ind]
    val_sample = np.concatenate((val_inl_sample, val_out_sample))

    val_inl_labels = np.ones(val_inl.shape[0]).astype(int)
    val_out_labels = np.zeros(val_out.shape[0]).astype(int)

    val_imgs = np.concatenate((val_inl, val_out))
    val_imgs = val_imgs / 127.5 - 1

    val_sample = val_sample / 127.5 - 1

    val_labels = np.concatenate((val_inl_labels, val_out_labels))

    print("Validation Shape", val_imgs.shape)

    val_imgs = tf.constant(val_imgs.astype("float32"))
    val_labels = tf.constant(val_labels.astype("float32"))

    return train_ds, val_imgs, val_labels, val_sample

def set_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def clean_dirs(dir_list):
    for dir in dir_list:
        if os.path.exists(dir) == True:
            os.system(" rm -r {}".format(dir))

        os.makedirs(dir)
