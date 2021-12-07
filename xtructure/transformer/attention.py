import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def attention_matrix(K, Q, use_mask=False):
    window_size_queries = Q.get_shape()[1]
    window_size_keys = K.get_shape()[1]
    mask = tf.convert_to_tensor(value=np.transpose(np.tril(np.ones((window_size_queries,window_size_keys))*np.NINF,-1),(1,0)),dtype=tf.float32)
    atten_mask = tf.tile(tf.reshape(mask,[-1,window_size_queries,window_size_keys]),[tf.shape(input=K)[0],1,1])

    attention = (Q @ tf.transpose(K, perm=[0, 2, 1])) / tf.sqrt(tf.cast(K.shape[-1], tf.float32))
    if use_mask:
        attention = attention + atten_mask
    return tf.nn.softmax(attention, axis=-1)


class SingleAttentionHead(layers.Layer):
    def __init__(self, input_size, output_size, use_mask):
        super().__init__()

        self.use_mask = use_mask
        self.Q = self.add_weight(name="Q", shape=(input_size, output_size), initializer="normal")
        self.K = self.add_weight(name="K", shape=(input_size, output_size), initializer="normal")
        self.V = self.add_weight(name="V", shape=(input_size, output_size), initializer="normal")

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries, training=False):
        Q = inputs_for_queries @ self.Q
        K = inputs_for_keys @ self.K
        V = inputs_for_values @ self.V
        attention = attention_matrix(K, Q, self.use_mask)
        return attention @ V


class AttentionHead(layers.Layer):
    def __init__(self, embedding_size, use_mask, num_heads=3):
        super().__init__()
        
        self.num_heads = num_heads
        output_size = embedding_size // num_heads
        self.heads = [SingleAttentionHead(embedding_size, output_size, use_mask) for _ in range(num_heads)]
        self.W = self.add_weight("W", shape=(output_size * num_heads, embedding_size), initializer="normal")

    @tf.function
    def call(self, inputs_keys, inputs_values, inputs_queries, training=False):
        Z = tf.concat([
            head(inputs_keys, inputs_values, inputs_queries, training=training)
            for head in self.heads
        ], axis=-1)
        return Z @ self.W


class TransformerBlock(layers.Layer):
    def __init__(self, embedding_size, is_decoder):
        super().__init__()

        self.dense0 = layers.Dense(embedding_size, activation="relu")
        self.dense1 = layers.Dense(embedding_size)
        self.self_attention = AttentionHead(embedding_size, use_mask=is_decoder)
        self.is_decoder = is_decoder
        if self.is_decoder:
            self.self_context_attention = AttentionHead(embedding_size, use_mask=False)

        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

    @tf.function
    def call(self, inputs, context=None, training=False):
        attention_out = self.self_attention(inputs, inputs, inputs, training=training)
        attention_out += inputs
        attention_normalized = self.layer_norm(attention_out)

        if self.is_decoder:
            context_attention_out = self.self_context_attention(context,context,attention_normalized, training=training)
            context_attention_out += attention_normalized
            attention_normalized = self.layer_norm(context_attention_out)

        ff_out = self.dense0(attention_normalized)
        ff_out = self.dense1(ff_out)
        ff_out += attention_normalized
        ff_norm = self.layer_norm(ff_out)

        return tf.nn.relu(ff_norm)


class PositionEncoding(layers.Layer):
    def __init__(self, window_size, embedding_size):
        super().__init__()
        self.denses = [
            layers.Dense(embedding_size)
            for i in range(window_size)
        ]
        self.window_size = window_size
        # self.positional_embeddings = self.add_weight("PE", shape=[window_sz, embedding_size])

    @tf.function
    def call(self, embedded, iam, training=False):
        """
        :param embedded: [batch_size, window_size, embedding_size]
        :param iam: [batch_size, iam_size]
        """
        positional_embeddings = tf.concat([
            self.denses[i](iam)[:, None, :]
            for i in range(self.window_size)
        ], axis=1)

        return embedded + positional_embeddings

# class PositionEncoding(layers.Layer):
#     def __init__(self, window_sz, embedding_size):
#         super().__init__()
#         self.positional_embeddings = self.add_weight("PE", shape=[window_sz, embedding_size])

#     @tf.function
#     def call(self, x):
#         return x + self.positional_embeddings
