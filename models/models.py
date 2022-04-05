import torch
from torch import nn
from torchvision import models as torch_models
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from tensorflow import keras


# from PathoMCH
def inception_keras(c):
    from inception import InceptionV3
    c.network_name = 'inception'
    model_input = keras.layers.Input(shape=(c.IMG_SIZE, c.IMG_SIZE, c.N_CHANNELS))
    base_model = InceptionV3(shape=(c.IMG_SIZE, c.IMG_SIZE, c.N_CHANNELS),
                             input_tensor=model_input,
                             include_top=False)
    x = base_model.output
    x = keras.layers.GlobalMaxPool2D()(x)
    out_activation = keras.layers.Dense(1, activation='sigmoid', name='sigmoid_activation_2class')(x)

    model = keras.models.Model(inputs=base_model.input, outputs=out_activation)
    return model

# from GeneMutationFromHE
class ResNet_extractor(nn.Module):
    def __init__(self, layers=101):
        super().__init__()
        if layers == 18:
            self.resnet = torch_models.resnet18(pretrained=True)
        elif layers == 34:
            self.resnet = torch_models.resnet34(pretrained=True)
        elif layers == 50:
            self.resnet = torch_models.resnet50(pretrained=True)
        elif layers == 101:
            self.resnet = torch_models.resnet101(pretrained=True)
        else:
            raise(ValueError('Layers must be 18, 34, 50 or 101.'))

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
def customized_conv_block(num_filter, size_filter, input_layer, batch_norm = False, dropout=0.2):
    conv = keras.layers.Conv2D(filters=num_filter, kernel_size=size_filter, padding="same")(input_layer)
    dropped = keras.layers.Dropout(dropout)(conv)
    if batch_norm:
        batched = keras.layers.BatchNorm2d()(dropped)
        activated = keras.layers.Activation('relu')(batched)
    else:
        activated = keras.layers.Activation('relu')(dropped)
    return activated
    
    
def customized_group_one_model(input_layer, normalizer, filters=[16, 32, 64], batch=False, dropout=0.2, num_classes):
    x = normalizer(input_layer)
    for f in filters:
        x = customized_conv_block(f, 3, x, batch, dropout)
    x = keras.layers.Flatten()(x)
    # into dense layers
    x = keras.layers.Dense(256)(x)
    x = keras.layers.Dense(128)(x)
    x = keras.layers.Dense(64)(x)
    output_layer = keras.layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = keras.Model(inputs=x, outputs=output_layer)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['mean_squared_error'])
    
    return model


def transferred_learning_model(input_layer, normalizer, input_shape, num_classes):
    base_model = keras.applications.Xception(weights='imagenet', input_shape=input_shape, include_top=False) 
    base_model.trainable = False
    inputs = normalizer(input_layer)
    # Pre-trained Xception weights requires that input be scaled
    # from (0, 255) to a range of (-1., +1.), the rescaling layer
    # outputs: `(inputs * scale) + offset`
    scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(inputs)
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['mean_squared_error'])
    
    return model