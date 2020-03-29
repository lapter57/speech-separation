import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Lambda, ReLU, Reshape, LSTM
from tensorflow.keras.layers import TimeDistributed, Bidirectional, BatchNormalization, concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.initializers import he_normal, glorot_uniform

def conv2d(filters, kernel_size, dilation_rate, name, model):
    model.add(Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), 
                  padding='same', dilation_rate=dilation_rate, name=name))
    model.add(BatchNormalization())
    model.add(ReLU())
    return model

def build_audio_conv_layers():
    model = Sequential()
    model = conv2d(96, (1, 7), (1, 1), 'a_conv1', model)
    model = conv2d(96, (7, 1), (1, 1), 'a_conv2', model)
    model = conv2d(96, (5, 5), (1, 1), 'a_conv3', model)
    model = conv2d(96, (5, 5), (2, 1), 'a_conv4', model)
    model = conv2d(96, (5, 5), (4, 1), 'a_conv5', model)
    model = conv2d(96, (5, 5), (8, 1), 'a_conv6', model)
    model = conv2d(96, (5, 5), (16, 1), 'a_conv7', model)
    model = conv2d(96, (5, 5), (32, 1), 'a_conv8', model)
    model = conv2d(96, (5, 5), (1, 1), 'a_conv9', model)
    model = conv2d(96, (5, 5), (2, 2), 'a_conv10', model)
    model = conv2d(96, (5, 5), (4, 4), 'a_conv11', model)
    model = conv2d(96, (5, 5), (8, 8), 'a_conv12', model)
    model = conv2d(96, (5, 5), (16, 16), 'a_conv13', model)
    model = conv2d(96, (5, 5), (32, 32), 'a_conv14', model)
    model = conv2d(8, (1, 1), (1, 1), 'a_conv15', model)
    return model

def build_video_conv_layers():
    model = Sequential()
    model = conv2d(256, (7, 1), (1, 1), 'v_conv1', model)
    model = conv2d(256, (5, 1), (1, 1), 'v_conv2', model)
    model = conv2d(256, (5, 1), (2, 1), 'v_conv3', model)
    model = conv2d(256, (5, 1), (4, 1), 'v_conv4', model)
    model = conv2d(256, (5, 1), (8, 1), 'v_conv5', model)
    model = conv2d(256, (5, 1), (16, 1), 'v_conv6', model)
    return model

def UpSampling2DBilinear(size):
    return Lambda(lambda x: tf.compat.v1.image.resize(x, size, align_corners=True))

def sliced(x, index):
    return x[:, :, :, index]

def build_ao_model(n_speakers):
    model = build_audio_conv_layers()
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(400, input_shape=(298, 8 * 257), return_sequences=True), merge_mode='sum'))
    model.add(Dense(600, name="fc1", activation='relu',
                kernel_initializer=he_normal(seed=27)))
    model.add(Dense(600, name="fc2", activation='relu',
                kernel_initializer=he_normal(seed=42)))
    model.add(Dense(600, name="fc3", activation='relu',
                kernel_initializer=he_normal(seed=65)))
    model.add(Dense(257 * 2 * n_speakers, name="complex_mask",
                         kernel_initializer=glorot_uniform(seed=87)))
    model.add(Reshape((298, 257, 2, n_speakers)))

    model_input = Input(shape=(298, 257, 2))
    AO_model = Model(inputs=model_input, outputs=model(model_input))
    AO_model.compile(optimizer=optimizers.Adam(), loss='mse')
    return AO_model

def build_av_model(n_speakers):
    a_model = build_audio_conv_layers()
    a_model.add(Reshape((298, 8 * 257)))
    a_input = Input(shape=(298, 257, 2))
    a_out = a_model(a_input)

    v_model = build_video_conv_layers()
    v_model.add(Reshape((75, 256, 1)))
    v_model.add(UpSampling2DBilinear((298, 256)))
    v_model.add(Reshape((298, 256)))

    v_input = Input(shape=(75, 1, 1792, n_speakers))
    AVfusion_list = [a_out]
    for i in range(n_speakers):
        single_input = Lambda(sliced, arguments={'index': i})(v_input)
        v_out = v_model(single_input)
        AVfusion_list.append(v_out)
    
    AVfusion = concatenate(AVfusion_list, axis=2)
    AVfusion = TimeDistributed(Flatten())(AVfusion)
    lstm = Bidirectional(LSTM(400, input_shape=(298, 8 * 257), return_sequences=True), merge_mode='sum')(AVfusion)
    fc1 = Dense(600, name="fc1", activation='relu', kernel_initializer=he_normal(seed=27))(lstm)
    fc2 = Dense(600, name="fc2", activation='relu', kernel_initializer=he_normal(seed=42))(fc1)
    fc3 = Dense(600, name="fc3", activation='relu', kernel_initializer=he_normal(seed=65))(fc2)
    complex_mask = Dense(257 * 2 * n_speakers, name="complex_mask", kernel_initializer=glorot_uniform(seed=87))(fc3)
    complex_mask_out = Reshape((298, 257, 2, n_speakers))(complex_mask)
    AV_model = Model(inputs=[a_input, v_input], outputs=complex_mask_out)
    AV_model.compile(optimizer='adam', loss='mse')
    return AV_model