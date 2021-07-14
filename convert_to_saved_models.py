import tensorflow as tf

model = tf.keras.models.load_model('reshaped_model.hdf5', compile = False)
tf.saved_model.save(model, './saved_models')
