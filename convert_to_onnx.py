import keras
import tensorflow as tf
from keras2onnx import convert_keras
import tensorflow as tf
import onnx
import tf2onnx.convert
from keras.optimizers import Adam
from tensorflow.keras import backend as K

'''
def dice_soft(y_true, y_pred, smooth=0.00001):
    # Identify axis
    axis = identify_axis(y_true.get_shape())

    # Calculate required variables
    intersection = y_true * y_pred
    intersection = K.sum(intersection, axis=axis)
    y_true = K.sum(y_true, axis=axis)
    y_pred = K.sum(y_pred, axis=axis)

    # Calculate Soft Dice Similarity Coefficient
    dice = ((2 * intersection) + smooth) / (y_true + y_pred + smooth)

    # Obtain mean of Dice & return result score
    dice = K.mean(dice)
    return dice

def tversky_loss(y_true, y_pred, smooth=0.000001):
    # Define alpha and beta
    alpha = 0.5
    beta  = 0.5
    # Calculate Tversky for each class
    axis = identify_axis(y_true.get_shape())
    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum(y_true * (1-y_pred), axis=axis)
    fp = K.sum((1-y_true) * y_pred, axis=axis)
    tversky_class = (tp + smooth)/(tp + alpha*fn + beta*fp + smooth)
    # Sum up classes to one score
    tversky = K.sum(tversky_class, axis=[-1])
    # Identify number of classes
    n = K.cast(K.shape(y_true)[-1], 'float32')
    # Return Tversky
    return n-tversky

def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')
'''
model = tf.keras.models.load_model('./reshaped_model.hdf5', compile = False)

onnx_model, _ = tf2onnx.convert.from_keras(model, opset = 12)

onnx.save(onnx_model, './model.onnx')


#def keras_to_onnx(model, output_filename):
#   onnx = convert_keras(model, output_filename)
#   with open(output_filename, "wb") as f:
#       f.write(onnx.SerializeToString())

#model = keras.models.load_model('./model.hdf5', custom_objects = {'custom_loss':tversky_loss, 'custom_metrics':dice_soft})
#model.compile(optimizer=Adam(lr=0.0001),
#                           loss=tversky_loss, metrics=[dice_soft])

#keras_to_onnx(model, './model.onnx') 

