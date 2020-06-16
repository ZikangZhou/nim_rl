# Copyright 2020 Zhou Zikang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import Input, layers, losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from pynim import *

Transition = collections.namedtuple(
    "Transition",
    "after_state reward next_state done")


class ReplayBuffer(object):
    def __init__(self, replay_buffer_capacity):
        self._replay_buffer_capacity = replay_buffer_capacity
        self._data = []
        self._next_entry_index = 0

    def add(self, element):
        if len(self._data) < self._replay_buffer_capacity:
            self._data.append(element)
        else:
            self._data[int(self._next_entry_index)] = element
            self._next_entry_index += 1
            self._next_entry_index %= self._replay_buffer_capacity

    def sample(self, num_samples):
        if len(self._data) < num_samples:
            raise ValueError(
                "{} elements could not be sampled from size {}".format(
                    num_samples, len(self._data)))
        return random.sample(self._data, num_samples)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


class DecayEpsilonGreedy(Exploration):
    def __init__(self,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_duration=int(1e6)):
        Exploration.__init__(self)
        self._epsilon = epsilon_start
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay_duration = epsilon_decay_duration

    def clone(self):
        cloned = DecayEpsilonGreedy.__new__(DecayEpsilonGreedy)
        Exploration.__init__(cloned, self)
        cloned.__dict__.update(self.__dict__)
        return cloned

    def explore(self, legal_actions, greedy_actions):
        if np.random.rand() < self._epsilon:
            return sample_action(legal_actions)
        else:
            return sample_action(greedy_actions)

    def update(self, episode):
        decay_steps = min(episode, self._epsilon_decay_duration)
        self._epsilon = self._epsilon_end + (
                self._epsilon_start - self._epsilon_end) * (
                                1 - decay_steps / self._epsilon_decay_duration)


class StepCounter(object):
    def __init__(self):
        self._counter = 0

    def count(self):
        self._counter += 1

    def get(self):
        return self._counter


class DQN(RLAgent):
    def __init__(self,
                 hidden_layers_size=2,
                 replay_buffer_capacity=10000,
                 batch_size=32,
                 replay_buffer_class=ReplayBuffer,
                 learning_rate=0.01,
                 update_target_network_every=100,
                 learn_every=1,
                 discount_factor=1.0,
                 min_buffer_size_to_learn=512,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_duration=int(10000),
                 optimizer_str="adam",
                 loss_str="mse"):
        RLAgent.__init__(self)

        self._hidden_layer_size = hidden_layers_size
        self._batch_size = batch_size
        self._replay_buffer = replay_buffer_class(replay_buffer_capacity)
        self._learning_rate = learning_rate
        self._update_target_network_every = update_target_network_every
        self._learn_every = learn_every
        self._min_buffer_size_to_learn = min_buffer_size_to_learn
        self._discount_factor = discount_factor

        self._epsilon_greedy = DecayEpsilonGreedy(epsilon_start, epsilon_end,
                                                  epsilon_decay_duration)

        self._step_counter = StepCounter()

        self._network = self.build_network()
        self._target_values = dict()

        if loss_str == "mse":
            self._loss_class = losses.mean_squared_error
        elif loss_str == "huber":
            self._loss_class = losses.Huber
        else:
            raise ValueError("Not implemented, choose from 'mse', 'huber'.")
        if optimizer_str == "adam":
            self._optimizer = optimizers.Adam(self._learning_rate)
        elif optimizer_str == "sgd":
            self._optimizer = optimizers.SGD(self._learning_rate)
        else:
            raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")

        self._all_states = []

    # def build_network(self):
    #     input1 = Input(shape=(2,))
    #     input2 = Input(shape=(2,))
    #     input3 = Input(shape=(2,))
    #     input4 = Input(shape=(2,))
    #     dense1 = layers.Dense(self._hidden_layer_size, activation='relu',
    #                           input_shape=(2,), dtype='float32')
    #     dense2 = layers.Dense(1, activation='sigmoid', dtype='float32')
    #     bit1 = dense2(dense1(input1))
    #     bit2 = dense2(dense1(input2))
    #     bit3 = dense2(dense1(input3))
    #     bit4 = dense2(dense1(input4))
    #     merged = layers.concatenate([bit1, bit2, bit3, bit4])
    #     x = layers.Dense(2, activation="relu")(merged)
    #     output_tensor = layers.Dense(1)(x)
    #     model = Model([input1, input2, input3, input4], output_tensor)
    #     model.summary()
    #     plot_model(model, show_shapes=True, to_file='model.png')
    #     return model

    @staticmethod
    def build_network():
        input1 = Input(shape=(2,))
        input2 = Input(shape=(2,))
        input3 = Input(shape=(2,))
        input4 = Input(shape=(2,))
        reshape = layers.Reshape((2, 1), dtype='float32')
        rnn = layers.SimpleRNN(1, input_shape=(2, 1), activation=tf.sin)
        bit1 = rnn(reshape(input1))
        bit2 = rnn(reshape(input2))
        bit3 = rnn(reshape(input3))
        bit4 = rnn(reshape(input4))
        merged = layers.concatenate([bit1, bit2, bit3, bit4])
        x = layers.Dense(2, activation="relu")(merged)
        output_tensor = layers.Dense(1)(x)
        model = Model([input1, input2, input3, input4], output_tensor)
        model.summary()
        plot_model(model, show_shapes=True, to_file='model.png')
        return model

    # @staticmethod
    # def build_network():
    #     input_tensor = Input(shape=(2, 4))
    #     x = layers.Conv1D(filters=4, kernel_size=2, activation=tf.sin,
    #                       dtype='float32')(input_tensor)
    #     x = layers.Flatten()(x)
    #     x = layers.Dense(4, activation=None)(x)
    #     x = layers.Dense(2, activation="relu")(x)
    #     output_tensor = layers.Dense(1)(x)
    #     model = Model(input_tensor, output_tensor)
    #     model.summary()
    #     plot_model(model, show_shapes=True, to_file='model.png')
    #     return model

    def clone(self):
        cloned = DQN.__new__(DQN)
        RLAgent.__init__(cloned, self)
        cloned.__dict__.update(self.__dict__)
        return cloned

    @staticmethod
    def feature(state):
        input_tensor = np.zeros((len(state), 4))
        for pile_id in range(len(state)):
            pile_bin = format(state[pile_id], '04b')
            for bit in range(4):
                input_tensor[pile_id][bit] = float(pile_bin[bit])
        return input_tensor

    @staticmethod
    def bit_feature(state, bit):
        input_tensor = np.zeros(len(state))
        for pile_id in range(len(state)):
            pile_bin = format(state[pile_id], '04b')
            input_tensor[pile_id] = float(pile_bin[bit])
        return input_tensor

    def predict_single(self, state):
        return self._network.predict(
            [self.bit_feature(state, bit)[np.newaxis, :] for bit in range(4)])[
            0][0]
        # return self._network.predict(self.feature(state)[np.newaxis, :])[0][0]

    def get_values(self):
        values = dict()
        for state in self._all_states:
            if state.is_terminal():
                values[state] = 1.0
            else:
                values[state] = self.predict_single(state)
        return values

    def initialize(self, all_states):
        self._all_states = all_states
        self._target_values = self.get_values()

    def policy(self, state, is_evaluation):
        legal_actions = state.legal_actions()
        self.set_legal_actions(legal_actions)
        self.clear_greedy_actions()
        if len(legal_actions) == 0:
            self.set_greedy_value(0.0)
            return Action()
        else:
            greedy_action = max(legal_actions, key=lambda a: 1.0 if state.child(
                a).is_terminal() else self.predict_single(state.child(a)))
            self.set_greedy_actions([greedy_action])
            if state.child(greedy_action).is_terminal():
                self.set_greedy_value(1.0)
            else:
                self.set_greedy_value(
                    self.predict_single(state.child(greedy_action)))

            if is_evaluation:
                return sample_action(self.get_greedy_actions())
            else:
                return self.policy_impl(legal_actions,
                                        self.get_greedy_actions())

    def policy_impl(self, legal_actions, greedy_actions):
        return self._epsilon_greedy.explore(legal_actions, greedy_actions)

    def step(self, game, is_evaluation):
        reward = TIE_REWARD
        done = False
        if game.is_terminal():
            reward = LOSE_REWARD
            done = True
        next_state = game.get_state()
        action = super().step(game, is_evaluation)
        if not is_evaluation:
            self._step_counter.count()
            if not self.get_current_state().is_empty() and \
                    not next_state.is_empty():
                transition = Transition(after_state=self.get_current_state(),
                                        reward=reward, next_state=next_state,
                                        done=done)
                self._replay_buffer.add(transition)
            self.set_current_state(game.get_state())
            if self._step_counter.get() % self._learn_every == 0:
                self.update(self.get_current_state(), game.get_state(), reward)
            if self._step_counter.get() % \
                    self._update_target_network_every == 0:
                self._target_values = self.get_values()
        return action

    def update(self, update_state, current_state, reward):
        if (len(self._replay_buffer) < self._batch_size or len(
                self._replay_buffer) < self._min_buffer_size_to_learn):
            return
        transitions = self._replay_buffer.sample(self._batch_size)
        after_states = [t.after_state for t in transitions]
        rewards = np.array([t.reward for t in transitions])
        next_states = [t.next_state for t in transitions]
        dones = np.array([t.done for t in transitions])

        greedy_next_values = np.zeros(len(next_states))
        for i in range(len(next_states)):
            legal_actions = next_states[i].legal_actions()
            if len(legal_actions) == 0:
                greedy_next_values[i] = 0.0
            else:
                greedy_next_action = max(legal_actions,
                                         key=lambda a: self._target_values[
                                             next_states[i].child(a)])
                greedy_next_after_state = next_states[i].child(
                    greedy_next_action)
                greedy_next_values[i] = self._target_values[
                    greedy_next_after_state]
        target_values = rewards + (
                1 - dones) * self._discount_factor * greedy_next_values
        # target_values = np.array(
        #     [1.0 if after_state.nim_sum() == 0 else -1.0 for after_state in
        #      after_states])

        target_values = tf.convert_to_tensor(target_values[:, np.newaxis])
        with tf.GradientTape() as tape:
            # values = self._network(np.array(
            #     [self.feature(after_state) for after_state in after_states]))
            input1 = np.array(
                [self.bit_feature(after_state, 0) for after_state in
                 after_states])
            input2 = np.array(
                [self.bit_feature(after_state, 1) for after_state in
                 after_states])
            input3 = np.array(
                [self.bit_feature(after_state, 2) for after_state in
                 after_states])
            input4 = np.array(
                [self.bit_feature(after_state, 3) for after_state in
                 after_states])
            values = self._network([input1, input2, input3, input4])
            loss = tf.reduce_mean(self._loss_class(target_values, values))
            grads = tape.gradient(loss, self._network.trainable_variables)
            self._optimizer.apply_gradients(
                zip(grads, self._network.trainable_variables))

    def update_exploration(self, episode):
        self._epsilon_greedy.update(episode)

    def get_weights(self):
        return self._network.get_weights()


def main():
    dqn = DQN()
    game = Game(State([15, 15]))
    game.set_first_player(dqn)
    game.set_second_player(dqn)
    game.train(10000)
    print(dqn.get_weights())
    game.print_values()
    game.set_second_player(OptimalAgent())
    game.play(10000)


if __name__ == '__main__':
    main()
