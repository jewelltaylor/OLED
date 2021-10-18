

import os
import gc
import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys

PERC = 87.5
BATCH_SIZE = 2000
MODEL_BASE_PATH = sys.argv[1]
MODEL_NAME = sys.argv[2]
TRAIN_IMG_PATH = sys.argv[3]
TRAIN_LABEL_PATH = sys.argv[4]
TEST_IMG_PATH = sys.argv[5]
TEST_LABEL_PATH = sys.argv[6]
SCORE_STR_LIST = ["auc_normal", "auc_mask", "auc_avg", "auc_context"]
SEED = 17
PERC = 87.5
TRAIN_PERC = .6667
OUTLIER_PERC = .30

os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
initializer = tf.keras.initializers.GlorotNormal(seed=SEED)

def get_auc(pred, labels, m):
    m.update_state(labels, pred)
    auc = m.result()
    return auc

def get_data(inlier):
    x_train = np.load(TRAIN_IMG_PATH).reshape((-1, 28, 28, 1))
    y_train = np.load(TRAIN_LABEL_PATH)
    x_test = np.load(TEST_IMG_PATH).reshape((-1, 28, 28, 1))
    y_test = np.load(TEST_LABEL_PATH)
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    x = np.concatenate((x_train, x_test)).astype("float32")
    x = x / 127.5 - 1
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

    return test_imgs, test_labels

def test(model_name, score_list):

    result_dict = {}

    for i in range(10):
        result_dict[str(i)] = []

    for i in range(10):
        print("Class: {}".format(str(i)))
        imgs, labels = get_data(i)
        data = tf.data.Dataset.from_tensor_slices((imgs.astype("float32")))
        data = data.batch(BATCH_SIZE)
        for score_str in score_list:
            model_class_name =  "{}_train_{}".format(model_name, str(i))
            base_path = os.path.join(MODEL_BASE_PATH, model_class_name, "{}_" + "{}.h5".format(score_str))
            decoder_normal_output, decoder_mask_output, missing_parts, generated_missing_parts, z_mean = [], [], [], [], []
            for n, (img) in data.enumerate():
                dno, dmo, mp, gmp, zm = test_step(img, n, base_path)
                decoder_normal_output.append(dno)
                decoder_mask_output.append(dmo)
                missing_parts.append(mp)
                generated_missing_parts.append(gmp)
                z_mean.append(zm)


            decoder_normal_output = tf.concat(decoder_normal_output, axis = 0)
            decoder_mask_output = tf.concat(decoder_mask_output, axis = 0)
            missing_parts = tf.concat(missing_parts, axis = 0)
            generated_missing_parts = tf.concat(generated_missing_parts, axis =0)
            z_mean = tf.concat(z_mean, axis=0)


            #Get Normal Reconstruction Anomaly Score
            gen_l2_loss_normal_not_reduced = tf.abs(imgs - decoder_normal_output) ** 2
            reconstruction_loss_normal_reduced = tf.map_fn(fn=lambda x : tf.reduce_sum(x), elems=gen_l2_loss_normal_not_reduced)
            min_normal = tf.reduce_min(reconstruction_loss_normal_reduced)
            max_normal = tf.reduce_max(reconstruction_loss_normal_reduced)
            reconstruction_loss_normal_normalized = tf.math.add(tf.math.divide(tf.math.subtract(reconstruction_loss_normal_reduced, max_normal), tf.math.subtract(max_normal, min_normal)), 1)

            #Get Masked Reconstruction Anomaly Score
            gen_l2_loss_mask_not_reduced = tf.abs(imgs - decoder_mask_output) ** 2
            reconstruction_loss_mask_reduced = tf.map_fn(fn=lambda x : tf.reduce_sum(x), elems=gen_l2_loss_mask_not_reduced)
            min_mask = tf.reduce_min(reconstruction_loss_mask_reduced)
            max_mask = tf.reduce_max(reconstruction_loss_mask_reduced)
            reconstruction_loss_mask_normalized = tf.math.add(tf.math.divide(tf.math.subtract( reconstruction_loss_mask_reduced, max_mask), tf.math.subtract(max_mask, min_mask)), 1)

            #Get Context Loss
            context_loss_not_reduced = tf.abs(missing_parts - generated_missing_parts)
            context_loss_reduced = tf.map_fn(fn=lambda x : tf.reduce_sum(x), elems=context_loss_not_reduced)

            min_context = tf.reduce_min(context_loss_reduced)
            max_context = tf.reduce_max(context_loss_reduced)

            context_loss_normalized = tf.math.add(tf.math.divide(tf.math.subtract(context_loss_reduced, max_context), tf.math.subtract(max_context, min_context)), 1)

            avg_score = (reconstruction_loss_normal_normalized + reconstruction_loss_mask_normalized + context_loss_normalized) / len(score_list)

            score = None
            if (score_str == "auc_normal"):
                score = reconstruction_loss_normal_normalized
            elif (score_str == "auc_mask"):
                score = reconstruction_loss_mask_normalized
            elif (score_str == "auc_context"):
                score = context_loss_normalized
            elif (score_str == "auc_avg"):
                score = avg_score
            else:
                raise ValueError("Invalid Score String")


            #Get average anomaly score
            m = tf.keras.metrics.AUC(num_thresholds = 100)

            #Calculate AUC
            auc = get_auc(score, labels, m)
            result_str = "Result: {} : {} : {}".format(score_str, str(i), str(auc))
            print(result_str)
            result_dict[str(i)].append((score_str, auc))

            del decoder_normal_output, decoder_mask_output, missing_parts, generated_missing_parts, z_mean

        del data, labels

    return result_dict

def test_step(imgs, num, base_path):
    print("Iteration Number: {}".format(str(num.numpy())))

    encoder_path = base_path.format("encoder")
    decoder_path = base_path.format("deocder")
    mask_model_path = base_path.format("mask_model")

    encoder = tf.keras.models.load_model(encoder_path)
    decoder = tf.keras.models.load_model(decoder_path)
    mask_model =  tf.keras.models.load_model(mask_model_path)

    #Generate Normal Reconstruction
    z_mean  = encoder(imgs, training=False)
    decoder_normal_output = decoder(z_mean, training = False)

    #Generate Mask
    activation_maps = mask_model(imgs, training = False)
    thresh = tf.map_fn(
        fn=lambda x : tfp.stats.percentile(x, PERC, interpolation='lower', keepdims=True), elems=activation_maps
    ) + .000000000001
    rev_act_maps = tf.add(activation_maps  * -1, thresh)
    relu = tf.keras.activations.relu(rev_act_maps)
    mask = tf.math.divide(relu, relu + .000000000001)

    #Get Masked Images
    mask = tf.dtypes.cast(mask, tf.float32)
    imgs = tf.dtypes.cast(imgs, tf.float32)
    masked_imgs = tf.math.multiply(imgs, mask)

    #Get Maksed Reconstructions
    z_mean_mask  = encoder(masked_imgs, training=False)
    decoder_mask_output = decoder(z_mean_mask, training = False)
    masks_inverse = tf.math.equal(mask, tf.constant(0, dtype=tf.float32))
    masks_inverse = tf.dtypes.cast(masks_inverse, tf.float32)
    missing_parts = tf.math.multiply(imgs, masks_inverse)
    generated_missing_parts = tf.math.multiply(decoder_mask_output, masks_inverse)

    del encoder, decoder, mask_model
    gc.collect()

    return decoder_normal_output, decoder_mask_output, missing_parts, generated_missing_parts, z_mean

result_dict = test(MODEL_NAME, SCORE_STR_LIST)


res_list = [[] for i in range(len(SCORE_STR_LIST))]

for key in result_dict.keys():
    res = result_dict[key]
    for i, (lst, tup) in enumerate(zip(res_list, res)):
        res_list[i].append(tup[1])



res_list = [np.array(item).mean() for item in res_list]

for auc_str, auc in zip(SCORE_STR_LIST, res_list):
    print(auc_str, auc)
