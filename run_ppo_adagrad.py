#!/usr/bin/python3
import argparse
import gym
import os
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from algo.ppo import PPOTrain


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', default=0.995, type=float)
    parser.add_argument('--iteration', default=int(1e4), type=int)
    #CartPole-v1, Acrobot-v1, Pendulum-v0, HalfCheetah-v2, Hopper-v2, Walker2d-v2, Humanoid-v2
    parser.add_argument('--env', help='gym name', default='Acrobot-v1')
    #adagrad, rmsprop, adadelta, adam, cocob
    parser.add_argument('--optimizer', help='optimizer type name', default='adagrad')
    parser.add_argument('--lr', help='Learning Rate', default=0.1)
    parser.add_argument('--logdir', help='log directory', default='log/train/ppo')
    parser.add_argument('--savedir', help='save directory', default='trained_models/ppo')
    return parser.parse_args()


def main(args):
    #init directories
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)
    if not os.path.isdir(args.logdir + '/' + args.env):
        os.mkdir(args.logdir + '/' + args.env)
    if not os.path.isdir(args.logdir + '/' + args.env + '/' + args.optimizer):
        os.mkdir(args.logdir + '/' + args.env + '/' + args.optimizer)
    if not os.path.isdir(args.logdir + '/' + args.env + '/' + args.optimizer + '/lr_' + str(args.lr)):
        os.mkdir(args.logdir + '/' + args.env + '/' + args.optimizer + '/lr_' + str(args.lr))
    args.logdir = args.logdir + '/' + args.env + '/' + args.optimizer + '/lr_' + str(args.lr)
    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)
    if not os.path.isdir(args.savedir + '/' + args.env):
        os.mkdir(args.savedir + '/' + args.env)
    if not os.path.isdir(args.savedir + '/' + args.env + '/' + args.optimizer):
        os.mkdir(args.savedir + '/' + args.env + '/' + args.optimizer)
    args.savedir = args.savedir + '/' + args.env + '/' + args.optimizer

    #init classes
    env = gym.make(args.env)
    env.seed(0)
    ob_space = env.observation_space
    Policy = Policy_net('policy', env, args.env)
    Old_Policy = Policy_net('old_policy', env, args.env)
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma, _optimizer=args.optimizer, _lr=args.lr)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir, sess.graph)
        sess.run(tf.global_variables_initializer())
        obs = env.reset()
        reward = 0
        success_num = 0

        for iteration in range(args.iteration):
            observations = []
            actions = []
            v_preds = []
            rewards = []
            episode_length = 0
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                episode_length += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, v_pred = Policy.act(obs=obs, stochastic=True)

                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                observations.append(obs)
                actions.append(act)
                v_preds.append(v_pred)
                rewards.append(reward)

                next_obs, reward, done, info = env.step(act)

                if done:
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                    obs = env.reset()
                    reward = -1
                    break
                else:
                    obs = next_obs

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=episode_length)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            if iteration == (args.iteration-1):
                saver.save(sess, args.savedir+'/model'+str(args).lr+'.ckpt')
                print('Clear!! Model saved.')
                break

            gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)
            gaes = np.array(gaes).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) / gaes.std()
            rewards = np.array(rewards).astype(dtype=np.float32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            PPO.assign_policy_parameters()

            inp = [observations, actions, gaes, rewards, v_preds_next]

            print('iteration:', iteration, ',rewards:', sum(rewards))

            # train
            for epoch in range(6):
                # sample indices from [low, high)
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=32)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          gaes=sampled_inp[2],
                          rewards=sampled_inp[3],
                          v_preds_next=sampled_inp[4])

            summary = PPO.get_summary(obs=inp[0],
                                      actions=inp[1],
                                      gaes=inp[2],
                                      rewards=inp[3],
                                      v_preds_next=inp[4])

            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
