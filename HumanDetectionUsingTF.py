from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

def build_model(input_shape):
    inputs = L.Input(input_shape)
    print(inputs.shape)
    
    #backbone
    backbone = MobileNetV2(
            include_top = False,
            weights = "imagenet",
            input_tensor = inputs,
            alpha=1.0
        )
    #backbone.summary()
    
    #Detection Head
    #x = backbone.get_layer("block_13_expand_relu").output
    x = backbone.output
    #decreasing the number of channels
    x = L.Conv2D(256, kernel_size = 1, padding = "same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    print(x.shape)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(4, activation="sigmoid")(x)
    print(x.shape)
    
    #Model
    model = Model(inputs,x)
    return model

if __name__ == "__main__":
    input_shape = (512,512,3)
    build_model(input_shape)
    model = build_model(input_shape)
    model.summary()