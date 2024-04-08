from tensorflow.keras import layers as l
import tensorflow as tf


class TokenAndPositionEmbedding(l.Layer):
    def __init__(self, wlen, maxlen, vocab_size, embed_dim, name=None, trainable=True, dtype=None, dynamic=False):
        super(TokenAndPositionEmbedding, self).__init__(name=name, trainable=trainable, dtype=dtype, dynamic=dynamic)
        self.token_emb = l.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = l.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.pos_emb_w = l.Embedding(input_dim=wlen, output_dim=embed_dim)

        self.attention = l.Attention()

        self.llen = maxlen
        self.wlen = wlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def get_config(self):
        cfg = super().get_config()
        cfg['wlen'] = self.wlen
        cfg['maxlen'] = self.llen
        cfg['vocab_size'] = self.vocab_size
        cfg['embed_dim'] = self.embed_dim
        return cfg

    def call(self, x, *args, **kwargs):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = tf.expand_dims(positions, 0)
        positions = tf.expand_dims(positions, 0)
        positions = self.pos_emb(positions)

        wlen = tf.shape(x)[-2]
        positions_w = tf.range(start=0, limit=wlen, delta=1)
        positions_w = tf.expand_dims(positions_w, 0)
        positions_w = tf.expand_dims(positions_w, -1)
        positions_w = self.pos_emb_w(positions_w)


        mask = tf.expand_dims(tf.cast(x > 0,'float'), -1)

        x = self.token_emb(x)

        s = x + positions + positions_w


        return s
