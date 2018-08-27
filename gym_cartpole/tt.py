import numpy as np
from gym.envs.classic_control import CartPoleEnv
import tensorflow as tf


# tf.constant_initializer：常量初始化函数
#
# tf.random_normal_initializer：正态分布
#
# tf.truncated_normal_initializer：截取的正态分布
#
# tf.random_uniform_initializer：均匀分布
#
# tf.zeros_initializer：全部是0
#
# tf.ones_initializer：全是1
#
# tf.uniform_unit_scaling_initializer
class DeepQNetwork:
    def __init__(
            self,
            n_actions=2,  # env.action_space.n 有多少个行为可以选择
            n_features=4,  # (env.observation_space.shape[0] 状态的类型（数组类型）
            learning_rate=0.01,  # 学习效率
            reward_decay=0.9,  # 奖励衰减值
            e_greedy=0.9,  # 几率，多少几率取得最有解
            replace_target_iter=300,
            memory_size=500,  # 仓库的存储大小
            batch_size=32,  # 批次大小
            e_greedy_increment=None,  # 几率增长率
            output_graph=False,  # 是否打印图表
    ):
        # 2*4+2=10  #observation 返回长度为4的数组 还有一个下一个_observation ,再加上action 和reward 所以是10
        self.memory = np.zeros((500, 10));

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.memory_size = 500
        self.learn_step_counter = 0
        self.replace_target_iter = 300
        self.epsilon = 0.9
        self.cost_his = []

    def _build_net(self):
        # ------------------ 建立评估网络 ------------------

        self.s = tf.placeholder(tf.float32, [None, 4], name='s')  # 输入  observation 返回长度为4的数组
        self.q_target = tf.placeholder(tf.float32, [None, 2], name='Q_target')  # 为了计算偏差，2=有两个行为可以选择，输入值
        # 建立第一层神经
        c_names, n_l1, w_initializer, b_initializer = \
            ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
            tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        w1 = tf.get_variable('w1', [4, 10], initializer=w_initializer, collections=c_names)
        b1 = tf.get_variable('b1', [1, 10], initializer=b_initializer, collections=c_names)
        l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)  # 输出结果变成了（？,10）

        # 建立第二层神经元
        w2 = tf.get_variable('w2', [10, 2], initializer=w_initializer, collections=c_names)
        b2 = tf.get_variable('b2', [1, 2], initializer=b_initializer, collections=c_names)
        self.q_eval = tf.nn.relu(tf.matmul(l1, w2) + b2)  # 输出结果变成了（？,2）#预测值
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        self._train_op = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)





        # ------------------ 建立目标网络------------------
        target_net_params = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

        self.s_ = tf.placeholder(tf.float32, [None, 4], name='s_')  # input
        w3 = tf.get_variable('w3', [4, 10], initializer=w_initializer, collections=target_net_params)
        b3 = tf.get_variable('b3', [1, 10], initializer=b_initializer, collections=target_net_params)
        l3 = tf.nn.relu(tf.matmul(self.s_, w3) + b3)

        w4 = tf.get_variable('w4', [10, 2], initializer=w_initializer, collections=target_net_params)
        b4 = tf.get_variable('b4', [1, 2], initializer=b_initializer, collections=target_net_params)
        self.q_next = tf.matmul(l3, w4) + b4

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):  # 如果没有定义，就重新定义
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))  # 后面追加数组，变成了 长度为10
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        ## [ 0.01421105  0.03287232  0.02552669 -0.00107699] 变成 [[ 0.01421105  0.03287232  0.02552669 -0.00107699]]
        observation = observation[np.newaxis, :]

        if np.random.uniform() < 0.9:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, 2)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter > 500:
            sample_index = np.random.choice(500, size=32)  # 0-500之间选择32个数
        else:
            sample_index = np.random.choice(self.memory_counter, size=32)

        batch_memory = self.memory[sample_index, :]  # 长度为32
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -4:],  # 取出数组的后4个值 [[[0 1 2 3 4]]] ->[[1 2 3 4]]
                self.s: batch_memory[:, :4],  # # 取出数组的前4个值 [[[0 1 2 3 4]]] ->[[0 1 2 3]]
            })

        q_target = q_eval.copy()
        batch_index = np.arange(32, dtype=np.int32)
        eval_act_index = batch_memory[:, 4].astype(int)
        reward = batch_memory[:, 5]
        q_target[batch_index, eval_act_index] = reward + 0.9 * np.max(q_next, axis=1)  # 实际得到的值
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :4],
                                                self.q_target: q_target})
        print(self.cost)
        self.cost_his.append(self.cost)  # 记录 cost 误差

        # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + 0.0001 if self.epsilon > 1.0 else 1.0
        self.learn_step_counter += 1
