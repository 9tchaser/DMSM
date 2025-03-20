import tensorflow as tf
from tensorflow.keras import layers, models

def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value

def conv_layer(in_channels, out_channels, kernel_size, use_bias=True):
    kernel_size = _make_pair(kernel_size)
    padding = 'same'
    return layers.Conv2D(out_channels, kernel_size, padding=padding, use_bias=use_bias)

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        return layers.ReLU()
    elif act_type == 'lrelu':
        return layers.LeakyReLU(alpha=neg_slope)
    elif act_type == 'prelu':
        return layers.PReLU(shared_axes=[1, 2])
    else:
        raise NotImplementedError(f'activation layer [{act_type}] is not found')

def sequential(*args):
    return models.Sequential(args)

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size)
    return conv

class Conv3XC(tf.keras.layers.Layer):
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, bias=True, relu=False):
        super(Conv3XC, self).__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.stride = s
        self.has_relu = relu
        gain = gain1

        self.sk = tf.keras.layers.Conv2D(filters=c_out, kernel_size=1, padding='valid', strides=s, use_bias=bias)
        self.conv = models.Sequential([
            layers.Conv2D(filters=c_in * gain, kernel_size=1, padding='valid', use_bias=bias),
            layers.Conv2D(filters=c_out * gain, kernel_size=3, strides=s, padding='valid', use_bias=bias),
            layers.Conv2D(filters=c_out, kernel_size=1, padding='valid', use_bias=bias),
        ])

    def call(self, x, training=False):
        
        pad = 1
        x_pad = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode="CONSTANT", constant_values=0)
        out = self.conv(x_pad) + self.sk(x)
        if self.has_relu:
            out = tf.nn.leaky_relu(out, alpha=0.05)
        return out

class SPAB(layers.Layer):
    def __init__(self, in_channels, mid_channels=None, out_channels=None, use_bias=False):
        super(SPAB, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        #self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = layers.Activation('swish')
        self.act2 = activation('lrelu', neg_slope=0.1)

    def call(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        #out2 = self.c2_r(out1_act)
        #out2_act = self.act1(out2)

        out3 = self.c3_r(out1_act)

        sim_att = tf.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att

        return out, out1, sim_att

class SPAN(models.Model):
    def __init__(self, num_in_ch, num_out_ch, feature_channels=48, upscale=4, use_bias=True, img_range=255.):
        super(SPAN, self).__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch
        self.img_range = img_range

        self.conv_1 = Conv3XC(in_channels, feature_channels, gain1=2, s=1)
        self.block_1 = SPAB(feature_channels, use_bias=use_bias)
        #self.block_2 = SPAB(feature_channels, use_bias=use_bias)
        #self.block_3 = SPAB(feature_channels, use_bias=use_bias)
        #self.block_4 = SPAB(feature_channels, use_bias=use_bias)
        self.block_6 = SPAB(feature_channels, use_bias=use_bias)

        self.conv_cat = conv_layer(feature_channels * 4, feature_channels, kernel_size=1, use_bias=True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)

        self.upsampler = pixelshuffle_block(feature_channels, out_channels, upscale_factor=upscale)

    def call(self, x):
        x = x * self.img_range

        out_feature = self.conv_1(x)

        out_b5, _, att1 = self.block_1(out_feature)
        #out_b5, _, att2 = self.block_2(out_b1)
        #out_b5, _, att3 = self.block_3(out_b2)

        #out_b5, _, att4 = self.block_4(out_b3)
        out_b6, out_b5_2, att6 = self.block_6(out_b5)

        out_b6 = self.conv_2(out_b6)
        out = self.conv_cat(tf.concat([out_feature, out_b6, out_b5, out_b5_2], axis=-1))
        output = self.upsampler(out)

        return output

if __name__ == "__main__":
    input = tf.random.normal((1, 2, 256, 256))
    input = tf.transpose(input, [0, 2, 3, 1])
    model = SPAN(2, 2, upscale=1, feature_channels=48)

    output = model(input)
    print(output.shape)