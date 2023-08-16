import tensorflow as tf
from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.backend import ops

from static import ImageNetwork


class CNNFrameEncoder(tf.keras.layers.Layer):
    def __init__(self, cnn_encoder_network: str, trainable: bool = True, **kwargs):
        self.image_encoder = None
        self.cnn_encoder_network = cnn_encoder_network
        self.trainable = trainable
        if cnn_encoder_network == ImageNetwork.INCEPTION_V3.value:
            full_image_encoder_network = tf.keras.applications.InceptionV3(include_top=True, weights='imagenet')
            self.image_encoder = tf.keras.Model(full_image_encoder_network.inputs, full_image_encoder_network.layers[-2].output)
        elif cnn_encoder_network == ImageNetwork.RESNET_50.value:
            full_image_encoder_network = tf.keras.applications.ResNet50(include_top=True, weights='imagenet')
            self.image_encoder = tf.keras.Model(full_image_encoder_network.inputs,
                                                full_image_encoder_network.layers[-2].output)
        elif cnn_encoder_network == ImageNetwork.VGG_19.value:
            full_image_encoder_network = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
            self.image_encoder = tf.keras.Model(full_image_encoder_network.inputs,
                                                full_image_encoder_network.layers[-2].output)
        else:
            raise ValueError(f"{cnn_encoder_network} is not supported. Choose either "
                             f"{ImageNetwork.INCEPTION_V3.value} or {ImageNetwork.RESNET_50.value}")

        if not trainable:
            for layer in self.image_encoder.layers:
                layer.trainable = False

        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor):
        return self.image_encoder(inputs)

    def get_config(self):
        layer_config = super().get_config()
        layer_config["cnn_encoder_network"] = self.cnn_encoder_network
        layer_config["trainable"] = self.trainable
        return layer_config


class CNNVideoEncoder(tf.keras.layers.Layer):
    def __init__(self, cnn_encoder_network: str, cnn_trainable: bool = True):
        self.cnn_frame_encoder = CNNFrameEncoder(cnn_encoder_network=cnn_encoder_network, trainable=cnn_trainable)
        self.video_encoder = tf.keras.layers.TimeDistributed(self.cnn_frame_encoder)
        super().__init__()

    def call(self, inputs: tf.Tensor):
        assert len(inputs.shape) == 5
        return self.video_encoder(inputs)


class TransformerSelfAttention(tf.keras.layers.Layer):
    def __init__(self, emb_dims: int, heads: int = 1):
        self.emb_dims = emb_dims
        self.heads = heads
        self.pos_encoding = CustomSinePositionEncoding()
        self.multi_head_attn = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=emb_dims)
        super().__init__()

    def call(self, inputs: tf.Tensor):
        position_encodings = self.pos_encoding(inputs)
        video_encoding_with_position = inputs + position_encodings
        new_video_embeddings = self.multi_head_attn(video_encoding_with_position, video_encoding_with_position, video_encoding_with_position)

        return new_video_embeddings

    def get_config(self):
        layer_config = super().get_config()
        layer_config["emb_dims"] = self.emb_dims
        layer_config["heads"] = self.heads
        return layer_config


class TransformerCrossAttention(tf.keras.layers.Layer):
    def __init__(self, emb_dims: int, heads: int = 1):
        self.emb_dims = emb_dims
        self.heads = heads
        self.multi_head_attn = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=emb_dims)
        super().__init__()

    def call(self, input_1: tf.Tensor, input_2: tf.Tensor):
        cross_attn_enc, attn_weights = self.multi_head_attn(input_2, input_1, input_1, return_attention_scores=True)
        return cross_attn_enc, attn_weights

    def get_config(self):
        layer_config = super().get_config()
        layer_config["emb_dims"] = self.emb_dims
        layer_config["heads"] = self.heads
        return layer_config


def define_pair_video_encoder(pre_process_network: str, emb_size: int, cnn_trainable: bool=True):
    if pre_process_network == ImageNetwork.INCEPTION_V3.value:
        frame_size = 299
    elif pre_process_network == ImageNetwork.RESNET_50.value:
        frame_size = 224
    elif pre_process_network == ImageNetwork.VGG_19.value:
        frame_size = 224
    else:
        raise ValueError(f"{pre_process_network} is not supported. Choose either "
                         f"{ImageNetwork.INCEPTION_V3.value} or {ImageNetwork.RESNET_50.value}")

    global_avg_pooling = tf.keras.layers.GlobalAveragePooling1D()

    cnn_vid_encoder = CNNVideoEncoder(pre_process_network, cnn_trainable=cnn_trainable)
    transformer_self_attn = TransformerSelfAttention(emb_size)
    transformer_cross_attn = TransformerCrossAttention(emb_size)

    frames1 = tf.keras.Input(shape=(None, frame_size, frame_size, 3), ragged=True)
    frames2 = tf.keras.Input(shape=(None, frame_size, frame_size, 3), ragged=True)
    vid_enc1 = cnn_vid_encoder(frames1)
    vid_enc2 = cnn_vid_encoder(frames2)
    vid_self_attn_enc1 = transformer_self_attn(vid_enc1)
    vid_self_attn_enc2 = transformer_self_attn(vid_enc2)
    full_vid_enc2, full_vid_enc2_attn = transformer_cross_attn(vid_self_attn_enc1, vid_self_attn_enc2)
    full_vid_enc1, full_vid_enc1_attn = transformer_cross_attn(vid_self_attn_enc2, vid_self_attn_enc1)
    full_vid_enc2_avg = global_avg_pooling(full_vid_enc2)
    full_vid_enc1_avg = global_avg_pooling(full_vid_enc1)

    full_vid_enc2_norm = tf.nn.l2_normalize(full_vid_enc2_avg, axis=-1)
    full_vid_enc1_norm = tf.nn.l2_normalize(full_vid_enc1_avg, axis=-1)

    model = tf.keras.Model(inputs=(frames1, frames2), outputs=(full_vid_enc1_norm, full_vid_enc1_attn, full_vid_enc2_norm, full_vid_enc2_attn))
    return model


@keras_nlp_export("keras_nlp.layers.CustomSinePositionEncoding")
class CustomSinePositionEncoding(tf.keras.layers.Layer):
    def __init__(
        self,
        max_wavelength=10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength

    def call(self, inputs: tf.Tensor):
        shape = ops.shape(inputs)
        try:
            seq_length = shape[-2]
        except ValueError:
            return inputs
        hidden_size = shape[-1]
        position = ops.cast(ops.arange(seq_length), self.compute_dtype)
        min_freq = ops.cast(1 / self.max_wavelength, dtype=self.compute_dtype)
        timescales = ops.power(
            min_freq,
            ops.cast(2 * (ops.arange(hidden_size) // 2), self.compute_dtype)
            / ops.cast(hidden_size, self.compute_dtype),
        )
        angles = ops.expand_dims(position, 1) * ops.expand_dims(timescales, 0)
        # even indices are sine, odd are cosine
        cos_mask = ops.cast(ops.arange(hidden_size) % 2, self.compute_dtype)
        sin_mask = 1 - cos_mask
        # embedding shape is [seq_length, hidden_size]
        positional_encodings = (
            ops.sin(angles) * sin_mask + ops.cos(angles) * cos_mask
        )

        return ops.broadcast_to(positional_encodings, shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_wavelength": self.max_wavelength,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


def triplet_loss(input1: tf.Tensor, input2: tf.Tensor, margin=0.2):
    input2_t = tf.transpose(input2)
    cos_sim_mat = tf.matmul(input1, input2_t)

    diag = tf.linalg.diag_part(cos_sim_mat)
    mean_mat = tf.linalg.set_diag(cos_sim_mat, tf.zeros(cos_sim_mat.shape[0]))
    mean_neg = tf.math.reduce_sum(mean_mat, axis=-1) / (cos_sim_mat.shape[0] - 1)

    pos_mask = (tf.eye(cos_sim_mat.shape[0], cos_sim_mat.shape[0]) == 1) | (mean_mat > tf.expand_dims(diag, -1))
    pos_mask = tf.cast(pos_mask, dtype=tf.float32)

    closest_mat = mean_mat - 2 * pos_mask
    closest_neg = tf.math.reduce_max(closest_mat, axis=-1)

    triplet_loss_1 = tf.math.maximum(mean_neg - diag + margin, 0)
    triplet_loss_2 = tf.math.maximum(closest_neg - diag + margin, 0)

    loss = tf.math.reduce_mean(triplet_loss_1 + triplet_loss_2)
    return loss

