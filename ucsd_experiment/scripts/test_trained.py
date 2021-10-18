import sys
import os
import gc
import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

MODEL_BASE_PATH = sys.argv[1]
MODEL_NAME = sys.argv[2]


BATCH_SIZE = int(sys.argv[3])

TEST_INLIER_PATH = sys.argv[4]
TEST_OUTLIER_PATH = sys.argv[5]

SCORE_STR_LIST = ["auc_normal", "auc_mask", "auc_avg", "auc_context"]

SEED = 17
PERC = 87.5

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

def get_data():
    inl_data = np.load(TEST_INLIER_PATH)
    out_data = np.load(TEST_OUTLIER_PATH)

    print(inl_data.shape)
    print(out_data.shape)
    data = np.concatenate((inl_data, out_data))
    data = data / 127.5 - 1

    inl_labels = np.ones(inl_data.shape[0]).astype(int)
    out_labels = np.zeros(out_data.shape[0]).astype(int)
    labels = np.concatenate((inl_labels, out_labels))

    return data, labels

def test(imgs, labels):
    result_list = []
    data = tf.data.Dataset.from_tensor_slices((imgs.astype("float32")))
    data = data.batch(BATCH_SIZE)
    for score_str in SCORE_STR_LIST:
        base_path = os.path.join(MODEL_BASE_PATH, MODEL_NAME, "{}_" + "{}.h5".format(score_str))
        decoder_normal_output, decoder_mask_output, missing_parts, generated_missing_parts, z_mean = [], [], [], [], []
        for n, (img) in data.enumerate():
            dno, dmo, mp, gmp, zm = test_step(img, base_path)
            decoder_normal_output.append(dno)
            decoder_mask_output.append(dmo)
            missing_parts.append(mp)
            generated_missing_parts.append(gmp)
            z_mean.append(zm)


        decoder_normal_output = tf.concat(decoder_normal_output, axis = 0)
        decoder_mask_output = tf.concat(decoder_normal_output, axis = 0)
        missing_parts = tf.concat(missing_parts, axis = 0)
        generated_missing_parts = tf.concat(generated_missing_parts, axis =0)
        z_mean = tf.concat(z_mean, axis=0)


        #Get Normal Reconstruction Anomaly Score
        gen_l2_loss_normal_not_reduced = tf.abs(imgs - decoder_normal_output) ** 2
        reconstruction_loss_normal_reduced = tf.map_fn(fn=lambda x : tf.reduce_sum(x), elems=gen_l2_loss_normal_not_reduced) * -1
        min_normal = tf.reduce_min(reconstruction_loss_normal_reduced)
        max_normal = tf.reduce_max(reconstruction_loss_normal_reduced)
        reconstruction_loss_normal_normalized = tf.math.add(tf.math.divide(tf.math.subtract(reconstruction_loss_normal_reduced, max_normal), tf.math.subtract(max_normal, min_normal)), 1)

        #Get Masked Reconstruction Anomaly Score
        gen_l2_loss_mask_not_reduced = tf.abs(imgs - decoder_mask_output) ** 2
        reconstruction_loss_mask_reduced = tf.map_fn(fn=lambda x : tf.reduce_sum(x), elems=gen_l2_loss_mask_not_reduced) * -1
        min_mask = tf.reduce_min(reconstruction_loss_mask_reduced)
        max_mask = tf.reduce_max(reconstruction_loss_mask_reduced)
        reconstruction_loss_mask_normalized = tf.math.add(tf.math.divide(tf.math.subtract( reconstruction_loss_mask_reduced, max_mask), tf.math.subtract(max_mask, min_mask)), 1)

        #Get Context Loss
        context_loss_not_reduced = tf.abs(missing_parts - generated_missing_parts)
        context_loss_reduced = tf.map_fn(fn=lambda x : tf.reduce_sum(x), elems=context_loss_not_reduced) * -1

        min_context = tf.reduce_min(context_loss_reduced)
        max_context = tf.reduce_max(context_loss_reduced)

        context_loss_normalized = tf.math.add(tf.math.divide(tf.math.subtract(context_loss_reduced, max_context), tf.math.subtract(max_context, min_context)), 1)

        avg_score = (reconstruction_loss_normal_normalized + reconstruction_loss_mask_normalized + context_loss_normalized) / len(SCORE_STR_LIST)

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
        m = tf.keras.metrics.AUC(num_thresholds = 1000)

        score_eer = score.numpy()

        fpr, tpr, thresholds = roc_curve(labels, score_eer, pos_label=1)

        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        #Calculate AUC
        auc = get_auc(score, labels, m)
        result_str = "{}: AUC: {}  EER: {}".format(score_str, str(auc), str(eer))
        print(result_str)
        result_list.append((score_str, auc, eer))

    return result_list

def test_step(imgs, base_path):

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


    return decoder_normal_output, decoder_mask_output, missing_parts, generated_missing_parts, z_mean

data, labels = get_data()

result_list = test(data, labels)

print(result_list)



res_list = [[] for i in range(len(SCORE_STR_LIST))]


res_list = [np.array(item).mean() for item in res_list]

for auc_str, auc in zip(SCORE_STR_LIST, res_list):
    print(auc_str, auc)
