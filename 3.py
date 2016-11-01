import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
import cv2
import cv2.cv as cv




CNN_INPUT_WIDTH = 84
CNN_INPUT_HEIGHT = 84
CNN_INPUT_DEPTH = 1
SERIES_LENGTH = 4

REWARD_COFF = 3.0

INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.1
REPLAY_SIZE = 80000
BATCH_SIZE = 32
GAMMA = 0.99
ENV_NAME = 'Breakout-v0'
EPISODE = 20000
STEP = 800
TEST = 10


class ImageProcess():
	def ColorMat2GrayFlat(self, state):
		# state_output = tf.image.rgb_to_grayscale(state_input)
		# state_output = tf.image.crop_to_bounding_box(state_output, 34, 0, 160, 160)
		# state_output = tf.image.resize_images(state_output, 84, 84, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		# state_output = tf.squeeze(state_output)
		# return state_output

		height = state.shape[0]
		width = state.shape[1]
		nchannel = state.shape[2]

		sHeight = int(height * 0.5)
		sWidth = CNN_INPUT_WIDTH

		state_gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)

		state_graySmall = cv2.resize(state_gray, (sWidth, sHeight), interpolation=cv2.INTER_AREA)

		cnn_inputImg = state_graySmall[21:, :]
		# rstArray = state_graySmall.reshape(sWidth * sHeight)
		cnn_inputImg = cnn_inputImg.reshape((CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT, CNN_INPUT_DEPTH))
		# print cnn_inputImg.shape
		# cv2.imshow('Img', cnn_inputImg )
		# cv2.waitKey(0)
		return cnn_inputImg




class DQN():
	def __init__(self,env):
		self.imageProcess = ImageProcess()
		self.epsilon = INITIAL_EPSILON
		self.replay_buffer = deque()
		self.recent_history_queue =deque()
		self.action_dim = env.action_space.n
		self.state_dim = CNN_INPUT_HEIGHT * CNN_INPUT_WIDTH
		self.time_step = 0
		print env.action_space

		self.session = tf.InteractiveSession()
		self.create_network()
		# self.create_training_method()


		self.merged = tf.merge_all_summaries()
		self.summary_writer = tf.train.SummaryWriter('/path/to/logs', self.session.graph)

		self.session.run(tf.initialize_all_variables())


	def create_network(self):

		INPUT_DEPTH = SERIES_LENGTH
		FIRST_LAYER_PATCH_SIZE = 8
		FIRST_LAYER_STRIDE = 4
		FIRST_LAYER_NUM = 16
		SECOND_LAYER_PATCH_SIZE = 4
		SECOND_LAYER_STRIDE = 2
		SECOND_LAYER_NUM = 32

		THIRD_LAYER_NUM = 256

		self.input_layer = tf.placeholder( tf.uint8,[ None,CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT,INPUT_DEPTH ], name='status-input')
		self.action_input = tf.placeholder( tf.int32, [None, self.action_dim])
		self.y_input = tf.placeholder( tf.float32, [None])


		self.X = tf.to_float(self.input_layer)
		self.action_input = tf.to_float( self.action_input )

		conv1 = tf.contrib.layers.convolution2d(
			self.X,16,(8,8),activation_fn=tf.nn.relu,stride=(4,4))

		conv2 = tf.contrib.layers.convolution2d(
			conv1, 32, (4,4), activation_fn=tf.nn.relu, stride = (2,2))



		# Fully connected layers
		# tf.contrib.layers.flatten(conv3)
		flattened = tf.reshape( conv2, [ -1, 11 * 11 * 32 ] )

		W1 = self.get_weights( [ 11 * 11 * 32, self.action_dim ] )
		b1 = self.get_bias( [ self.action_dim ] )


		self.Q_value = tf.matmul( flattened, W1 ) + b1

		Q_action = tf.reduce_sum( tf.mul( self.Q_value, self.action_input ), reduction_indices = 1 )
		self.cost = tf.reduce_mean( tf.square( self.y_input - Q_action ) )

		self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.cost)
    #
	# def create_training_method(self):
    #
	# 	# if len(self.recent_history_queue) > 4:
	# 	# 	sess = tf.Session()
	# 	# 	print sess.run(self.Q_value)
	# 	# global_step = tf.Variable(0, name='global_step', trainable=True)
	# 	# self.optimizer = tf.train.AdamOptimizer( 0.001 ).minimize( self.cost )

	def train_network(self):
		self.time_step += 1


		minibatch = random.sample( self.replay_buffer, BATCH_SIZE )
		state_batch = [ data[ 0 ] for data in minibatch ]
		action_batch = [ data[ 1 ] for data in minibatch ]
		reward_batch = [ data[ 2 ] for data in minibatch ]
		next_state_batch = [ data[ 3 ] for data in minibatch ]
		done_batch = [ data[ 4 ] for data in minibatch ]

		y_batch = []
		Q_value_batch = self.Q_value.eval( feed_dict = { self.input_layer : next_state_batch } )
		for i in xrange( BATCH_SIZE ):
			if done_batch[i]:
				y_batch.append( reward_batch[ i ] )
			else:
				y_batch.append( reward_batch[ i ] + GAMMA * np.max(Q_value_batch[ i ])  )



		self.optimizer.run( feed_dict = {

			self.input_layer: state_batch,
			self.action_input:action_batch,
			self.y_input : y_batch

			} )




	def getRecentHistory_stack( self, state, append_or_not ): # get the state from environment and return the stack of 84 *84 *4 state_in_shadow if cout >  SERIES_LENGTH
		state_gray = self.imageProcess.ColorMat2GrayFlat( state )


		if append_or_not:
			self.recent_history_queue.append(state_gray)
			if len(self.recent_history_queue) > REPLAY_SIZE:
				self.recent_history_queue.popleft()

		if len(self.recent_history_queue) > SERIES_LENGTH:
			if append_or_not:
				state_of_shadow = np.dstack((
					self.recent_history_queue[-1],
					self.recent_history_queue[-2],
					self.recent_history_queue[-3],
					self.recent_history_queue[-4]
				))
			else:
				state_of_shadow = np.dstack((
					state_gray,
					self.recent_history_queue[-1],
					self.recent_history_queue[-2],
					self.recent_history_queue[-3]
				))


		else:
			state_of_shadow = []


		return state_of_shadow
	# def getRecentHistoryNext_stack(self, next_state): # get the state from next_state and return the 84 *84 *4
	# 	next_state_gray = self.imageProcess.ColorMat2GrayFlat( next_state )
	# 	state_of_shadow_next = np.dstack((
	# 		next_state_gray,
	# 		self.recent_history_queue[-1],
	# 		self.recent_history_queue[-2],
	# 		self.recent_history_queue[-3])
	# 	)
    #
	# 	return state_of_shadow_next
    #



	def percieve( self, state, action, reward, next_state, done, time_step ):
		one_hot_action = np.zeros( self.action_dim )
		one_hot_action[ action ] = 1

		state_of_shadow = self.getRecentHistory_stack( state, append_or_not= True )



		state_of_shadow_next = self.getRecentHistory_stack( next_state, append_or_not= False )


		if len(state_of_shadow) != 0:
			self.replay_buffer.append([state_of_shadow, one_hot_action, reward, state_of_shadow_next, done])



		if len(self.replay_buffer)> REPLAY_SIZE:
			self.replay_buffer.popleft()

		if len(self.replay_buffer) > BATCH_SIZE:
			self.train_network()


	def get_greedy_action( self, state ):

		rst = self.Q_value.eval( feed_dict = { self.input_layer : [state] } )[0]
		# print rst
		return np.argmax( rst )

	def get_action( self, state ):
		if self.epsilon >= FINAL_EPSILON:
			self.epsilon -= ( INITIAL_EPSILON - FINAL_EPSILON ) / 10000

		state_stack = self.getRecentHistory_stack( state, append_or_not= False )

		if random.random() < self.epsilon or len(state_stack) == 0:
			return random.randint( 0, self.action_dim - 1 )
		else:
			return self.get_greedy_action( state_stack )

	def get_weights( self, shape ):
		weight  = tf.truncated_normal(shape, stddev=0.01)
		return tf.Variable(weight)

	def get_bias( self, shape):
		bias = tf.constant( 0.01, shape = shape )
		return tf.Variable( bias )


def main():
	env = gym.make( ENV_NAME )


	agent = DQN( env )
	total_reward_decade = 0
	for episode in xrange(EPISODE):
		state = env.reset()
		total_reward = 0
		debug_reward = 0

		for step in xrange(STEP):
			env.render()
			action = agent.get_action( state )

			next_state, reward, done, _ = env.step( action )
			total_reward += reward

			agent.percieve( state, action, reward, next_state, done, episode )
			state = next_state

			if done:
				break
		print 'Episode:', episode, 'Total Point this Episode is:', total_reward
		total_reward_decade += total_reward
		if episode % 10 == 0:
			print '-------------'
			print 'Decade:', episode / 10, 'Total Reward in this Decade is:', total_reward_decade
			print '-------------'
			total_reward_decade = 0
		# summary = agent.session.run()
		# agent.summary_writer.add_summary( summary, episode )
		# print 'Episode:', episode, 'Debug Reward this Episode is:', debug_reward
		# if episode % 10 == 0:
		# 	total_reward = 0
		# 	for test in xrange( TEST ):
		# 		state = env.reset()
		# 		for step in xrange(STEP):
		# 			env.render()
		# 			action = agent.get_greedy_action( state )
		# 			next_state, reward, done, _ = env.step( action )
		# 			reward = reward * REWARD_COFF
		# 			state = next_state
		# 			total_reward += reward
		# 			if done:
		# 				break




if __name__ == '__main__':
	main()

