import tensorflow as tf
import algo.CocobOptimizer as cocob

class Discriminator:
    def __init__(self, env, env_name, _optimizer='adam'):
        """
        :param env:
        Output of this Discriminator is reward for learning agent. Not the cost.
        Because discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a).
        """
        self._optimizer = _optimizer
        env_header = env_name.split('-')[0]
        # CartPole-v1, Arcobot-v1, Pendulum-v0, HalfCheetah-v2, Hopper-v2, Walker2d-v2, Humanoid-v2
        if env_header == 'CartPole' or env_header == 'Arcobot' or env_header == 'Pendulum' or env_header == 'MountainCar': #Classic control Gym
            action_space_count = env.action_space.n
        else: #Mujoco
            action_space_count = env.action_space.shape[0]

        with tf.variable_scope('discriminator'):
            self.scope = tf.get_variable_scope().name
            self.expert_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.observation_space.shape))
            self.expert_a = tf.placeholder(dtype=tf.int32, shape=[None])
            expert_a_one_hot = tf.one_hot(self.expert_a, depth=action_space_count)
            # add noise for stabilise training
            expert_a_one_hot += tf.random_normal(tf.shape(expert_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            expert_s_a = tf.concat([self.expert_s, expert_a_one_hot], axis=1)

            self.agent_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.observation_space.shape))
            self.agent_a = tf.placeholder(dtype=tf.int32, shape=[None])
            agent_a_one_hot = tf.one_hot(self.agent_a, depth=action_space_count)
            # add noise for stabilise training
            agent_a_one_hot += tf.random_normal(tf.shape(agent_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            agent_s_a = tf.concat([self.agent_s, agent_a_one_hot], axis=1)

            with tf.variable_scope('network') as network_scope:
                prob_1 = self.construct_network(input=expert_s_a)
                network_scope.reuse_variables()  # share parameter
                prob_2 = self.construct_network(input=agent_s_a)

            with tf.variable_scope('loss'):
                loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(prob_1, 0.01, 1)))
                loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(1 - prob_2, 0.01, 1)))
                loss = loss_expert + loss_agent
                loss = -loss
                tf.summary.scalar('discriminator', loss)

            # optimizer: adagrad, rmsprop, adadelta, adam, cocob
            if self._optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)  # initial_accumulator_value=0.1
            elif self._optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025)  # decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, centered=False
            elif self._optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.5)  # learning_rate=0.001, rho=0.95, epsilon=1e-08, use_locking=False
            elif self._optimizer == 'cocob':
                optimizer = cocob.COCOB()
            else:  # adam
                optimizer = tf.train.AdamOptimizer()  # lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False
            self.train_op = optimizer.minimize(loss)

            self.rewards = tf.log(tf.clip_by_value(prob_2, 1e-10, 1))  # log(P(expert|s,a)) larger is better for agent

    def construct_network(self, input):
        layer_1 = tf.layers.dense(inputs=input, units=20, activation=tf.nn.leaky_relu, name='layer1')
        layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.nn.leaky_relu, name='layer2')
        layer_3 = tf.layers.dense(inputs=layer_2, units=20, activation=tf.nn.leaky_relu, name='layer3')
        prob = tf.layers.dense(inputs=layer_3, units=1, activation=tf.sigmoid, name='prob')
        return prob

    def train(self, expert_s, expert_a, agent_s, agent_a):
        return tf.get_default_session().run(self.train_op, feed_dict={self.expert_s: expert_s,
                                                                      self.expert_a: expert_a,
                                                                      self.agent_s: agent_s,
                                                                      self.agent_a: agent_a})

    def get_rewards(self, agent_s, agent_a):
        return tf.get_default_session().run(self.rewards, feed_dict={self.agent_s: agent_s,
                                                                     self.agent_a: agent_a})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

