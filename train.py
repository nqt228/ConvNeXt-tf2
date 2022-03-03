import tensorflow as tf
import tensorflow_datasets as tfds
from model.build import build_model
import os 




os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


ds_train, ds_val = tfds.load('imagenette', split=['train', 'validation'], shuffle_files=True,  as_supervised=True,)

def normalize_img(image, label):
  image = tf.cast(image, tf.float32) / 255.
  image = tf.image.resize(image, [300,300])
  print(image.shape)
  return image, label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.batch(18)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


ds_val = ds_val.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_val = ds_val.batch(18)
ds_val = ds_val.prefetch(tf.data.AUTOTUNE)


model = build_model('convnext_tiny', num_classes=10, input_shape = (300, 300, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
model.fit(ds_train, batch_size=32, epochs=10 , validation_data=ds_val)


