import tensorflow.keras as keras

model = keras.Sequential([keras.layers.Dense(64, activation=keras.activations.relu, input_shape=[2]),
	keras.layers.Dense(64, activation=keras.activations.relu),
	keras.layers.Dense(1)])

def custom_loss_function(y_actual, y_predicted):
	custom_loss_value = keras.backend.mean(keras.backend.sum(keras.backend.square((y_actual-y_predicted)/10)))
	return custom_loss_value

optimizer = keras.optimizers.RMSprop(0.001)
model.compile(loss=custom_loss_function, optimizer=optimizer)

model.summary()