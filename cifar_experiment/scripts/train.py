import os
import gc
import sys
import time
import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

NAME = sys.argv[1]
MODEL_SAVE_PATH = sys.argv[2] + "/" + NAME

INLIER = int(sys.argv[3])
EPOCHS = int(sys.argv[4])
BATCH_SIZE = int(sys.argv[5])

TRAIN_IMG_PATH = sys.argv[6]
TRAIN_LABEL_PATH = sys.argv[7]

IMG_SHAPE = (32, 32, 3)
SEED = 17
PERC = 87.5
LATENT_DIM = 256
#Setting seeds

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
initializer = tf.keras.initializers.GlorotNormal(seed=SEED)

dir_list = [MODEL_SAVE_PATH]

for dir in dir_list:
    if os.path.exists(dir) == True:
        os.system(" rm -r {}".format(dir))

    os.mkdir(dir)


def conv_block(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    use_bias=True,
    use_bn=False,
    use_dropout=False,
    drop_value=0.5,
):
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, kernel_initializer=initializer
    )(x)
    if use_bn:
        x =  tf.keras.layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x =  tf.keras.layers.Dropout(drop_value)(x)
    return x

def upsample_block(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(1, 1),
    up_size=(2, 2),
    padding="same",
    use_bn=False,
    use_bias=True,
    use_dropout=False,
    drop_value=0.3,
):
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, kernel_initializer=initializer
    )(x)

    x = tf.keras.layers.UpSampling2D(up_size)(x)

    if use_bn:
        x =  tf.keras.layers.BatchNormalization()(x)

    if activation:
        x = activation(x)
    if use_dropout:
        x =  tf.keras.layers.Dropout(drop_value)(x)
    return x

def get_encoder_model():
    img_input =  tf.keras.layers.Input(shape=IMG_SHAPE)

    x = conv_block(
        img_input,
        128,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=True,
        use_bias=True,
        activation=  tf.keras.layers.LeakyReLU(.2),
    )

    x = conv_block(
        x,
        256,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=True,
        use_bias=True,
        activation=  tf.keras.layers.LeakyReLU(.2),
    )

    x = conv_block(
        x,
        512,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=True,
        activation= tf.keras.layers.LeakyReLU(.2),
        use_bias=True,
    )

    x = conv_block(
        x,
        768,
        kernel_size=(5, 5),
        strides=(1, 1),
        use_bn=True,
        activation= tf.keras.layers.LeakyReLU(.2),
        use_bias=True,
    )

    x = tf.keras.layers.Flatten()(x)

    z = tf.keras.layers.Dense(256, name="z")(x)

    enc_model = tf.keras.models.Model(inputs=img_input, outputs=z, name="generator")
    return enc_model

def get_decoder_model():
    latent =  tf.keras.layers.Input(shape=LATENT_DIM)
    x = tf.keras.layers.Dense(2048)(latent)
    x = tf.keras.layers.Reshape((4, 4, 128))(x)

    x = upsample_block(
        x,
        768,
        tf.keras.layers.LeakyReLU(.2),
        kernel_size=(5, 5),
        strides=(1, 1),
        up_size=(2, 2),
        use_bias=True,
        use_bn=False,
        padding="same",
    )

    x = upsample_block(
        x,
        512,
        tf.keras.layers.LeakyReLU(.2),
        kernel_size=(5, 5),
        strides=(1, 1),
        up_size=(2, 2),
        use_bias=True,
        use_bn=False,
        padding="same",
    )

    x = upsample_block(
        x,
        256,
        tf.keras.layers.LeakyReLU(.2),
        kernel_size=(5, 5),
        up_size=(2, 2),
        strides=(1, 1),
        use_bias=True,
        use_bn=False,
        padding="same",
    )

    x = conv_block(
        x,
        128,
        tf.keras.layers.LeakyReLU(.2),
        kernel_size=(5, 5),
        strides=(1, 1),
        use_bias=True,
        use_bn=False,
        padding="same",
    )

    x = conv_block(
        x,
        3,
        tf.keras.layers.LeakyReLU(.2),
        kernel_size=(5, 5),
        strides=(1, 1),
        use_bias=True,
        use_bn=False,
        padding="same",
    )

    dec_model = tf.keras.models.Model(inputs=latent, outputs=x, name="deocder")
    return dec_model

def reconstruction_loss(imgs, generator_output):
    gen_l2_loss = tf.reduce_mean(tf.abs(imgs - generator_output) ** 2)

    return gen_l2_loss

def get_context_loss(missing_parts, generated_missing_parts):
    context_loss = 30 * tf.reduce_mean(tf.abs(missing_parts - generated_missing_parts))
    return context_loss

def get_mask_model():
    img_input = tf.keras.layers.Input(shape=IMG_SHAPE)

    x = conv_block(
        img_input,
        32,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        use_bias=True,
        activation=tf.keras.layers.LeakyReLU(),
    )

    x = conv_block(
        x,
        48,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=True,
        activation=tf.keras.layers.LeakyReLU(),
        use_bias=True,
    )

    x = conv_block(
        x,
        64,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        use_bias=True,
        activation=tf.keras.layers.LeakyReLU(),
    )


    x = upsample_block(
        x,
        64,
        tf.keras.layers.LeakyReLU(),
        kernel_size=(5, 5),
        strides=(1, 1),
        up_size=(2, 2),
        use_bias=True,
        use_bn=False,
        padding="same",
    )

    x = upsample_block(
        x,
        48,
        tf.keras.layers.LeakyReLU(),
        kernel_size=(5, 5),
        strides=(1, 1),
        up_size=(2, 2),
        use_bias=True,
        use_bn=False,
        padding="same",
    )

    x = upsample_block(
        x,
        32,
        tf.keras.layers.LeakyReLU(),
        kernel_size=(5, 5),
        strides=(1, 1),
        up_size=(2, 2),
        use_bias=True,
        use_bn=False,
        padding="same",
    )

    x = upsample_block(
        x,
        1,
        tf.keras.layers.ReLU(),
        kernel_size=(5, 5),
        strides=(1, 1),
        up_size=(1, 1),
        use_bias=True,
        use_bn=False,
        padding="same",
    )


    mask_model = tf.keras.models.Model(img_input, x, name="mask")
    return mask_model

def sample(imgs, epoch):
    #Generate Normal Reconstruction
    z  = encoder(imgs, training=False)
    decoder_normal_output = decoder(z, training = False)

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
    z_mask  = encoder(masked_imgs, training=False)
    decoder_mask_output = decoder(z_mask, training = False)

    return decoder_mask_output.numpy(), decoder_normal_output.numpy(), masked_imgs.numpy()

def get_auc(discriminator_generated_output, validation_labels, m):
    y_hat = discriminator_generated_output
    m.update_state(validation_labels, y_hat)
    auc = m.result()
    return auc

@tf.function
def train_step(imgs, epoch):
    with tf.GradientTape() as enc_tape,  tf.GradientTape() as dec_tape, tf.GradientTape() as mask_tape:

        #Generate Normal Reconstruction
        z  = encoder(imgs, training=True)
        decoder_normal_output = decoder(z, training = True)

        #Generate Mask
        activation_maps = mask_model(imgs, training = True)
        thresh = tf.map_fn(
            fn=lambda x : tfp.stats.percentile(x, PERC, interpolation='lower', keepdims=True), elems=activation_maps
        ) + .000000000001
        rev_act_maps = tf.add(activation_maps  * -1, thresh)
        relu = tf.keras.activations.relu(rev_act_maps)

        #Get Masked Images
        mask = tf.math.divide(relu, relu + .000000000001)
        mask = tf.dtypes.cast(mask, tf.float32)
        imgs = tf.dtypes.cast(imgs, tf.float32)
        masked_imgs = tf.math.multiply(imgs, mask)

        #Get Maksed Reconstructions
        z_mask = encoder(masked_imgs, training=True)
        decoder_mask_output = decoder(z_mask, training = True)

        #Get Masked Region of Original and Generated Image
        masks_inverse = tf.math.equal(mask, tf.constant(0, dtype=tf.float32))
        masks_inverse = tf.dtypes.cast(masks_inverse, tf.float32)
        missing_parts = tf.math.multiply(imgs, masks_inverse)
        generated_missing_parts = tf.math.multiply(decoder_mask_output, masks_inverse)

        #Get Losses
        gen_l2_loss_normal = reconstruction_loss(imgs, decoder_normal_output)
        gen_l2_loss_mask = reconstruction_loss(imgs, decoder_mask_output)
        context_loss = get_context_loss(missing_parts, generated_missing_parts)
        total_encoder_loss = gen_l2_loss_normal + gen_l2_loss_mask + context_loss
        total_decoder_loss = gen_l2_loss_normal + gen_l2_loss_mask + context_loss
        mask_model_loss = gen_l2_loss_mask * -1

        #Calculate gradients and update weights
        encoder_gradients = enc_tape.gradient(total_encoder_loss, encoder.trainable_variables)
        encoder_optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))

        decoder_gradients = dec_tape.gradient(total_decoder_loss, decoder.trainable_variables)
        decoder_optimizer.apply_gradients(zip(decoder_gradients, decoder.trainable_variables))

        if epoch % 4 == 0:
            mask_model_gradients = mask_tape.gradient(mask_model_loss, mask_model.trainable_variables)
            mask_model_optimizer.apply_gradients(zip(mask_model_gradients, mask_model.trainable_variables))

        gc.collect()


@tf.function
def validation_step(validation_imgs, validation_labels, m1, m2, m3, m4, m5, epoch):

    #Generate Normal Reconstruction
    z  = encoder(validation_imgs, training=False)
    decoder_normal_output = decoder(z, training = False)

    #Generate Mask
    activation_maps = mask_model(validation_imgs, training = False)
    thresh = tf.map_fn(
        fn=lambda x : tfp.stats.percentile(x, PERC, interpolation='lower', keepdims=True), elems=activation_maps
    ) + .000000000001
    rev_act_maps = tf.add(activation_maps  * -1, thresh)
    relu = tf.keras.activations.relu(rev_act_maps)
    mask = tf.math.divide(relu, relu + .000000000001)

    #Get Masked Images
    mask = tf.dtypes.cast(mask, tf.float32)
    validation_imgs = tf.dtypes.cast(validation_imgs, tf.float32)
    masked_imgs = tf.math.multiply(validation_imgs, mask)

    #Get Maksed Reconstructions
    z_mask  = encoder(masked_imgs, training=False)
    decoder_mask_output = decoder(z_mask, training = False)

    #Get Masked Region of Original and Generated Image
    masks_inverse = tf.math.equal(mask, tf.constant(0, dtype=tf.float32))
    masks_inverse = tf.dtypes.cast(masks_inverse, tf.float32)
    missing_parts = tf.math.multiply(validation_imgs, masks_inverse)
    generated_missing_parts = tf.math.multiply(decoder_mask_output, masks_inverse)


    #Get Normal Reconstruction Anomaly Score
    gen_l2_loss_normal_not_reduced = tf.abs(validation_imgs - decoder_normal_output) ** 2
    reconstruction_loss_normal_reduced = tf.map_fn(fn=lambda x : tf.reduce_sum(x), elems=gen_l2_loss_normal_not_reduced) * -1
    min_normal = tf.reduce_min(reconstruction_loss_normal_reduced)
    max_normal = tf.reduce_max(reconstruction_loss_normal_reduced)
    reconstruction_loss_normal_normalized = tf.math.add(tf.math.divide(tf.math.subtract(reconstruction_loss_normal_reduced, max_normal), tf.math.subtract(max_normal, min_normal)), 1)

    #Get Masked Reconstruction Anomaly Score
    gen_l2_loss_mask_not_reduced = tf.abs(validation_imgs - decoder_mask_output) ** 2
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

    #Get average anomaly score
    avg_score = (reconstruction_loss_normal_normalized + reconstruction_loss_mask_normalized + context_loss_normalized) / 3

    #Get Validation Losses
    gen_l2_loss_normal = reconstruction_loss(validation_imgs, decoder_normal_output)
    gen_l2_loss_mask = reconstruction_loss(validation_imgs, decoder_mask_output)
    context_loss = get_context_loss(missing_parts, generated_missing_parts)
    total_encoder_loss = gen_l2_loss_normal + context_loss + gen_l2_loss_mask


    #Calculate AUC
    auc_rs = get_auc(reconstruction_loss_normal_normalized, validation_labels, m1)
    auc_rs2 = get_auc(reconstruction_loss_mask_normalized, validation_labels, m2)
    auc_rs4 = get_auc(avg_score, validation_labels, m4)
    auc_rs5 = get_auc(context_loss_normalized, validation_labels, m5)

    gc.collect()


    return auc_rs, auc_rs2, auc_rs4, auc_rs5


def fit(train_dataset, validation_images, validation_labels, val_sample, epochs):
  auc_str_list = ["auc_normal", "auc_mask", "auc_avg", "auc_context"]
  highest_auc_list = [0 for i in range(len(auc_str_list))]

  for epoch in range(epochs):

      start = time.time()
      print("Epoch: ", epoch + 1)

      # Train Loop
      for n, (imgs) in train_dataset.enumerate():
          train_step(imgs, epoch)

      #Objects to track AUC
      m1 = tf.keras.metrics.AUC(num_thresholds = 10)
      m2 = tf.keras.metrics.AUC(num_thresholds = 10)
      m3 = tf.keras.metrics.AUC(num_thresholds = 10)
      m4 = tf.keras.metrics.AUC(num_thresholds = 10)
      m5 = tf.keras.metrics.AUC(num_thresholds = 10)

      #Validation Step
      auc_rs, auc_rs2, auc_rs4, auc_rs5  = validation_step(validation_images, validation_labels, m1, m2, m3, m4, m5, epoch)

      auc_result_list = [auc_rs, auc_rs2, auc_rs4, auc_rs5]

      for i, (auc_str, highest_auc, auc) in enumerate(zip(auc_str_list, highest_auc_list, auc_result_list)):
          if auc > highest_auc:
              encoder.save(MODEL_SAVE_PATH + "/encoder_{}.h5".format(auc_str))
              decoder.save(MODEL_SAVE_PATH + "/deocder_{}.h5".format(auc_str))
              mask_model.save(MODEL_SAVE_PATH + "/mask_model_{}.h5".format(auc_str))

              highest_auc_list[i] = auc

      print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))
      tf.keras.backend.clear_session()
      gc.collect()


def get_data():
    x_train = np.load(TRAIN_IMG_PATH)
    y_train = np.load(TRAIN_LABEL_PATH)


    inl = x_train[y_train == INLIER].reshape((-1, 32, 32, 3))
    out = x_train[y_train != INLIER].reshape((-1, 32, 32, 3))

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

    val_inl_sample_ind = np.random.choice(range(val_inl.shape[0]), 100, replace=False)
    val_out_sample_ind = np.random.choice(range(val_out.shape[0]), 100, replace=False)
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

    return train_ds, val_imgs.astype("float32"), val_labels, val_sample.astype("float32")


train_dataset, val_imgs, val_labels, val_sample = get_data()
encoder = get_encoder_model()
encoder.summary()

decoder = get_decoder_model()
decoder.summary()

mask_model = get_mask_model()
mask_model.summary()

loss_object = tf.keras.losses.BinaryCrossentropy()

encoder_optimizer = tf.keras.optimizers.Adam(learning_rate = 5e-4,  beta_1=0.5, beta_2=0.9)
decoder_optimizer = tf.keras.optimizers.Adam(learning_rate = 5e-4,  beta_1=0.5, beta_2=0.9)
mask_model_optimizer = tf.keras.optimizers.Adam(learning_rate = 5e-4,  beta_1=0.5, beta_2=0.9)

fit(train_dataset, val_imgs, val_labels, val_sample, EPOCHS)
