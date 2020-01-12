from tensorflow.keras.layers import Input, Dense, Convolution2D, Bidirectional, BatchNormalization, ReLU, Reshape, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

def AO_model(n_speakers):
    model_input = Input(shape=(298, 257, 2))
    conv1 = Convolution2D(96, kernel_size=(1, 7), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv1')(model_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)

    conv2 = Convolution2D(96, kernel_size=(7, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv2')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)

    conv3 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv3')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)

    conv4 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(2, 1), name='conv4')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)

    conv5 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(4, 1), name='conv5')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = ReLU()(conv5)

    conv6 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(8, 1), name='conv6')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = ReLU()(conv6)

    conv7 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(16, 1), name='conv7')(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = ReLU()(conv7)
    
    conv8 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(32, 1), name='conv8')(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = ReLU()(conv8)

    conv9 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv9')(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = ReLU()(conv9)

    conv10 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(2, 2), name='conv10')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = ReLU()(conv10)

    conv11 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(4, 4), name='conv11')(conv10)
    conv11 = BatchNormalization()(conv11)
    conv11 = ReLU()(conv11)

    conv12 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(8, 8), name='conv12')(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = ReLU()(conv12)

    conv13 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(16, 16), name='conv13')(conv12)
    conv13 = BatchNormalization()(conv13)
    conv13 = ReLU()(conv13)
    
    conv14 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(32, 32), name='conv14')(conv13)
    conv14 = BatchNormalization()(conv14)
    conv14 = ReLU()(conv14)

    conv15 = Convolution2D(8, kernel_size=(1, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv15')(conv14)
    conv15 = BatchNormalization()(conv15)
    conv15 = ReLU()(conv15)

    AVfusion = Reshape((298, 8 * 257))(conv15)
    
    lstm = Bidirectional(LSTM(200, return_sequences=True))(AVfusion)

    fc1 = Dense(600, name="fc1", activation='relu', kernel_initializer=he_normal(seed=27))(lstm)

    fc2 = Dense(600, name="fc2", activation='relu', kernel_initializer=he_normal(seed=42))(fc1)

    fc3 = Dense(600, name="fc3", activation='relu', kernel_initializer=he_normal(seed=65))(fc2)

    complex_mask = Dense(257 * 2 * n_speakers, name="complex_mask", kernel_initializer=he_normal(seed=65))(fc3)

    complex_mask_out = Reshape((298, 257, 2, n_speakers))(complex_mask)
    
    AO_model = Model(inputs=model_input, outputs=complex_mask_out)
    AO_model.compile(optimizer=optimizers.Adam(), loss='mse')
    return AO_model