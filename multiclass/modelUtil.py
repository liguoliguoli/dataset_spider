import keras


def get_hsv_mlp():
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(72,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation="relu"))
    model.add(keras.layers.Dense(16))
    model.add(keras.layers.Activation(activation="softmax"))
    return model


def get_hu_mlp():
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(7,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation="relu"))
    model.add(keras.layers.Dense(16))
    model.add(keras.layers.Activation(activation="softmax"))
    return model


def get_lbp_mlp():
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(26,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation="relu"))
    model.add(keras.layers.Dense(16))
    model.add(keras.layers.Activation(activation="softmax"))
    return model


def get_hog_mlp():
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(3600,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Activation(activation="relu"))
    model.add(keras.layers.Dense(16))
    model.add(keras.layers.Activation(activation="softmax"))
    return model


def get_resnet50_mlp():
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(2048,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Activation(activation="relu"))
    model.add(keras.layers.Dense(16))
    model.add(keras.layers.Activation(activation="softmax"))
    return model


def get_text_mlp():
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(100,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation="relu"))
    model.add(keras.layers.Dense(16))
    model.add(keras.layers.Activation(activation="softmax"))
    return model


def get_text_embeding(input_dim, output_dim, maxlen):
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=maxlen))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(16))
    model.add(keras.layers.Activation(activation="softmax"))
    return model


def get_text_lstm(input_dim, output_dim, maxlen):
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=maxlen))
    model.add(keras.layers.LSTM(units=32))
    model.add(keras.layers.Dense(16))
    model.add(keras.layers.Activation(activation="softmax"))
    return model


def get_composed_model(models, trainable=False):
    "composed model for superlearn"
    if not trainable:
        for i, model in enumerate(models):
            model.trainable = False
            for layer in model.layers:
                layer.name = layer.name + "_model%s"%i
    inputs = [model.input for model in models]
    out_puts = [model.output for model in models]
    x = keras.layers.concatenate(out_puts, axis=-1)
    x = keras.layers.core.Reshape((1, 16, len(models)))(x)
    x = keras.layers.Conv2D(16, 1, activation="relu")(x)
    x = keras.layers.Flatten()(x)
    out = keras.layers.Dense(16, activation="softmax")(x)
    model = keras.models.Model(inputs, out)
    return model