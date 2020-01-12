from tensorflow.keras.layers import Input, Dense, Conv2D, Bidirectional, BatchNormalization, ReLU, Reshape, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

def conv2d(filters, kernel_size, dilation_rate, name, model):
    conv = Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), 
                          padding='same', dilation_rate=dilation_rate, name=name)(model)
    conv = BatchNormalization()(conv)
    return ReLU()(conv)

def AO_model(n_speakers):
    model_input = Input(shape=(298, 257, 2))
    conv1 = conv2d(96, (1, 7), (1, 1), 'conv1', model_input)
    conv2 = conv2d(96, (7, 1), (1, 1), 'conv2', conv1)
    conv3 = conv2d(96, (5, 5), (1, 1), 'conv3', conv2)
    conv4 = conv2d(96, (5, 5), (2, 1), 'conv4', conv3)
    conv5 = conv2d(96, (5, 5), (4, 1), 'conv5', conv4)
    conv6 = conv2d(96, (5, 5), (8, 1), 'conv6', conv5)
    conv7 = conv2d(96, (5, 5), (16, 1), 'conv7', conv6)
    conv8 = conv2d(96, (5, 5), (32, 1), 'conv8', conv7)
    conv9 = conv2d(96, (5, 5), (1, 1), 'conv9', conv8)
    conv10 = conv2d(96, (5, 5), (2, 2), 'conv10', conv9)
    conv11 = conv2d(96, (5, 5), (4, 4), 'conv11', conv10)
    conv12 = conv2d(96, (5, 5), (8, 8), 'conv12', conv11)
    conv13 = conv2d(96, (5, 5), (16, 16), 'conv13', conv12)
    conv14 = conv2d(96, (5, 5), (32, 32), 'conv14', conv13)
    conv15 = conv2d(8, (1, 1), (1, 1), 'conv15', conv14)
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