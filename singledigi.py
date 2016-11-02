""" An example showing how to save/restore models and retrieve weights. """

from __future__ import absolute_import, division, print_function
from PIL import Image
import numpy
from numpy import newaxis
import tflearn
from tflearn.utils import feed_dict_builder
import pdb
import tflearn.datasets.mnist as mnist

# # MNIST Data
X, Y, testX, testY = mnist.load_data(one_hot=True)

# # Model
# input_layer = tflearn.input_data(shape=[None, 784], name='input')
# dense1 = tflearn.fully_connected(input_layer, 128, name='dense1')
# dense2 = tflearn.fully_connected(dense1, 256, name='dense2')
# softmax = tflearn.fully_connected(dense2, 10, activation='softmax')
# regression = tflearn.regression(softmax, optimizer='adam',
#                                 learning_rate=0.001,
#                                 loss='categorical_crossentropy')
#
# # # Define classifier, with model checkpoint (autosave)
# model = tflearn.DNN(regression, checkpoint_path='model.ckpt')
#
# # Train model, with model checkpoint every epoch and every 500 training steps.
# model.fit(X, Y, n_epoch=1,
#           validation_set=(testX, testY),
#           show_metric=True,
#           snapshot_epoch=True,  # Snapshot (save & evaluate) model every epoch.
#           # Snapshot (save & evalaute) model every 500 steps.
#           snapshot_step=500,
#           run_id='model_and_weights')
#
#
# # # ---------------------
# # # Save and load a model
# # # ---------------------
#
# # # Manually save model
# model.save("model.tfl")
#
# # Load a model
# model.load("model.tfl")

# Or Load a model from auto-generated checkpoint
# >> model.load("model.tfl.ckpt-500")

# # Resume training
# model.fit(X, Y, n_epoch=1,
#           validation_set=(testX, testY),
#           show_metric=True,
#           snapshot_epoch=True,
#           run_id='model_and_weights')


# # ------------------
# # Retrieving weights
# # ------------------

# # Retrieve a layer weights, by layer name:
# dense1_vars = tflearn.variables.get_layer_variables_by_name('dense1')
# # Get a variable's value, using model `get_weights` method:
# print("Dense1 layer weights:")
# print(model.get_weights(dense1_vars[0]))
# # Or using generic tflearn function:
# print("Dense1 layer biases:")
# with model.session.as_default():
#     print(tflearn.variables.get_value(dense1_vars[1]))

# # It is also possible to retrieve a layer weights through its attributes `W`
# # and `b` (if available).
# # Get variable's value, using model `get_weights` method:
# print("Dense2 layer weights:")
# print(model.get_weights(dense2.W))
# # Or using generic tflearn function:
# print("Dense2 layer biases:")
# with model.session.as_default():
#     print(tflearn.variables.get_value(dense2.b))

# filename = "/Users/wasprobot/workspace/ml/softmax/images/8.png"
# image = Image.open(filename)
# image_arr = numpy.divide(numpy.mod(numpy.asarray(image), 255), 255.0).ravel()
# print(numpy.argmax(model.predict([image_arr])))

# print(numpy.shape(X[0].T))
# print(numpy.shape(Y))

colorindex = numpy.unique(X.ravel())
colorindex = dict(enumerate(colorindex))

input_layer = tflearn.input_data(
    shape=[None, 784, len(colorindex)], name='input')
dense1 = tflearn.fully_connected(input_layer, 128, name='dense1')
dense2 = tflearn.fully_connected(dense1, 256, name='dense2')
softmax = tflearn.fully_connected(
    dense2, len(colorindex), activation='softmax')
regression = tflearn.regression(softmax, optimizer='adam',
                                learning_rate=0.001,
                                loss='categorical_crossentropy')

m = tflearn.SequenceGenerator(regression, dictionary=colorindex,
                              seq_maxlen=784,
                              clip_gradients=5.0)
pxarray = []
for px in X[0]:
    for key, val in colorindex.items():
        if val == px:
            pxarray.append(key)

for i in range(784):
    x = numpy.zeros((1, 784, 256))
    for t, char in enumerate(pxarray):
        x[0, t, colorindex[char]]=1.
    feed_dict=feed_dict_builder(x, None, m.inputs, None)
    print(m.predictor.predict(feed_dict)[0])

# print(m.generate(784, temperature=1.2, seq_seed=pxarray))
