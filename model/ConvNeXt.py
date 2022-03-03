import tensorflow as tf 
from tensorflow.keras.layers import Conv2D,LayerNormalization,ReLU,DepthwiseConv2D,Dense,GlobalAveragePooling2D, Input
from tensorflow.keras import Sequential
import tensorflow_addons as tfa 



kernel_initial = tf.keras.initializers.TruncatedNormal(stddev=0.2)
bias_initial = tf.keras.initializers.Constant(value=0)

class Block(tf.keras.layers.Layer):
	def __init__(self, dim, drop_prob=0, ):
		super().__init__()
		self.dwconv = DepthwiseConv2D(kernel_size = (7,7),padding = 'same',
			kernel_initializer=kernel_initial, bias_initializer=bias_initial)
		# self.dwconv = Conv2D(dim,kernel_size = (7,7),padding = 'same',groups=dim,
		# 	kernel_initializer=kernel_initial, bias_initializer=bias_initial)
		self.norm = LayerNormalization(epsilon=1e-6)
		self.pwconv1 = Conv2D(dim*4, kernel_size=1, padding='valid',
			kernel_initializer=kernel_initial, bias_initializer=bias_initial)
		self.act = tfa.layers.GELU()
		self.pwconv2 = Conv2D(dim, kernel_size=1, padding = 'valid',
			kernel_initializer=kernel_initial, bias_initializer=bias_initial)
		self.stochastic_depth = tfa.layers.StochasticDepth(drop_prob)
	def call(self,x):
		input = x 
		x = self.dwconv(x)
		x = self.norm(x)
		x = self.pwconv1(x)
		x = self.act(x)
		x = self.pwconv2(x)
		x = self.stochastic_depth([input , x ])

		return x 

class ConvNeXt(tf.keras.Model):
	def __init__(self,num_classes = 1000, depths = [3, 3, 9, 3], dims=[96, 192, 384, 768], input_shape = (32, 32, 3)):
		super().__init__()
		self.downsample_layers = []
		stem  = Sequential([
			Conv2D(dims[0], kernel_size=4, strides=4, padding='valid', input_shape = input_shape),
	 		LayerNormalization(epsilon=1e-6,axis=1)]
	 		)
		self.downsample_layers.append(stem)
		for i in range(3):
			downsample_layer = Sequential([
				LayerNormalization(epsilon=1e-6, axis=1),
				Conv2D(dims[i+1],kernel_size=2,strides=2,padding='valid')]
				)
			self.downsample_layers.append(downsample_layer)
		self.stages = []
		for i in range(4):
			stage = Sequential(
				[*[Block(dim=dims[i]) for j in range(depths[i])]])
			self.stages.append(stage)
		self.norm = LayerNormalization(epsilon=1e-6)
		self.head = Dense(num_classes)
		self.GAP  = GlobalAveragePooling2D()
	def call_features(self,x):
		for i in range(4):
			x = self.downsample_layers[i](x)
			x = self.stages[i](x)
		return self.norm(self.GAP(x))
	def call(self,x):
		x = self.call_features(x)
		x = self.head(x)
		return x




def convnext_tiny(num_classes):
	model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=num_classes)
	return model


def convnext_small(num_classes):
	model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], num_classes=num_classes)
	return model


def convnext_base(num_classes):
	model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], num_classes=num_classes)
	return model


def convnext_large(num_classes):
	model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], num_classes=num_classes)
	return model


def convnext_xlarge(num_classes):
	model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], num_classes=num_classes)
	return model




# a = tf.random.uniform(shape=[1,300,300,3], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None)
# block =  Block(dim=3)
# out = block(a)

# net = ConvNeXt()
# out = net(a)
# print(out.shape)


# model = convnext_tiny(num_classes=100)
# out = model(a)
# print(out.shape)