#coding: utf-8
"""
Environment is a Robot Arm. The arm tries to get to the blue point.
The environment will return a geographic (distance) information for the arm to learn.

The far away from blue point the less reward; touch blue r+=1; stop at blue for a while then get r=+10.

You can train this RL by using LOAD = False, after training, this model will be store in the a local folder.
Using LOAD = True to reload the trained model for playing.

You can customize this script in a way you want.

View more on [莫烦Python] : https://morvanzhou.github.io/tutorials/

Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1
tensorflow >= 1.0.1
"""

import tensorflow as tf
import numpy as np
import os
import shutil
from Env import ENV
import random


np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODES = 500   #default 600
MAX_EP_STEPS = 500
LR_A = 3e-4  # learning rate for actor
LR_C = 3e-4  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 1100
REPLACE_ITER_C = 1000
MEMORY_CAPACITY = 5000
BATCH_SIZE = 64
VAR_MIN = 0.1
RENDER = True
LOAD = False
NUM_AGENTS=1
NUM_OBSTACLES=6
AGENTS_RADIUS=10
viewer_xy=(400,400)
MODE = ['easy', 'hard']
n_model = 1



env = ENV(NUM_AGENTS,NUM_OBSTACLES,AGENTS_RADIUS,viewer_xy)
STATE_DIM = env.state_dim#7
ACTION_DIM = env.action_dim#2
ACTION_BOUND = env.action_bound#[-1,1]
#tf. get_variable()变量共享，它会去搜索变量名，然后没有就新建，有就直接用
# all placeholder for tf
with tf.name_scope('S'):#命名域，会给操作加上前缀
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')#创建输入维度，刚开始不包含任何状态表示
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')#从一个结合中取出全部变量，是一个列表
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')
#name_scope 是给op_name加前缀, variable_scope是给get_variable()创建的变量的名字加前缀
    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):#共享变量，允许变量名一样，也是命名域，不管是否有操作有加上前缀
            init_w = tf.contrib.layers.xavier_initializer()#一种经典的权值矩阵的初始化方式;
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 200, activation=tf.nn.relu6,#dense全连接层
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])#tf.assign(A, new_number): 这个函数的功能主要是把A的值变为new_number
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :] # single state插入新的维度的意思 加上偏置
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)#grad_ys权重，梯度求导值乘以权重，q对动作的梯度

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):#a_=actor.a_
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1
        #print("learn:",self.t_replace_counter)

class Memory(object):
    def __init__(self, capacity, dims):#capacity存储量
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))#融合成一列
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


sess = tf.Session()

# Create actor and critic.
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)#reward discount
actor.add_grad_to_graph(critic.a_grads)#self.a_grads = tf.gradients(self.q, a)[0] 奖励对动作进行求导

M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

saver = tf.train.Saver()
path = './'+MODE[n_model]

if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())


def train():
    var = 0.5  # control exploration
    expore_rate=1

    for ep in range(MAX_EPISODES):#600个片段
        if ep%10==0 and ep!=0:
            expore_rate*=0.9
        s = env.reset()#1，state 
        ep_reward = [0]*NUM_AGENTS
        #print("s:",s)
        #expore_rate=max(0.1,expore_rate*0.999)
        print("expore_rate:",expore_rate)

        done=[False]*NUM_AGENTS
        for t in range(MAX_EP_STEPS):#200一个片段两百次尝试
            if RENDER:
                env.render()
            
            random_actions=env.sample_action()
            
            for idx in range(NUM_AGENTS):
                a = actor.choose_action(s[idx])
                
                if random.randint(0,100)<expore_rate*100:
                    a=random_actions[idx]
                    #print("random_a:",a)
                #else:
                #a = np.clip(np.random.normal(a, var), *ACTION_BOUND)
                #print("a:",a,"expore_rate:",var)
                
                if done[idx]==True:
                    continue
                
                s_, r, done_i = env.step_single(a,idx)
                
                #print("a",a,"r",r)
                
                done[idx]=done_i
                #print("done:",done)
                M.store_transition(s[idx], a, r, s_)
                
                if M.pointer > MEMORY_CAPACITY:#5000
                    var = max([var*.9999, VAR_MIN]) # decay the action randomness
                    b_M = M.sample(BATCH_SIZE)#16 随机采集16组数据
                    b_s = b_M[:, :STATE_DIM]#状态数据
                    b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]#动作数据
                    b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]#奖励数据
                    b_s_ = b_M[:, -STATE_DIM:]#下一个状态
                    critic.learn(b_s, b_a, b_r, b_s_)
                    actor.learn(b_s)
                s[idx]=s_
                ep_reward[idx] += r
            
            if t == MAX_EP_STEPS-1 :
                result = '| done' if done else '| ----'
                print('Ep:', ep,
                  result,
                  '| R: %i' % int(ep_reward[idx]),
                  '| Explore: %.2f' % expore_rate,
                  )
                break
                

    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join('./'+MODE[n_model], 'DDPG.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)


def eval():
    env.set_fps(30)
    s = env.reset()
    steps_num=0
    while True:
        steps_num+=1
        if steps_num>300:
            s = env.reset()
            steps_num=0
        done=[False]*NUM_AGENTS
        if RENDER:
            env.render()
        for i in range(NUM_AGENTS):
            if done[i]:
                continue
            a = actor.choose_action(s[i])#训练结束后只需使用动作旋转网络进行动作决策。
            print("a:",a)
        #input()
            s_, r, done_i = env.step_single(a,i)
        #print("r:",r)
            s[i] = s_
            done[i]=done_i
            if done==[True]*len(done):
                s = env.reset()
            #print("len:",len(s))

if __name__ == '__main__':
    if LOAD:
        eval()
    else:
        train()
