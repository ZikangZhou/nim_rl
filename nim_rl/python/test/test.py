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

from pynim import *

if __name__ == '__main__':
    game = Game(State([10, 10, 10]))
    human_agent = HumanAgent()
    optimal_agent = OptimalAgent()
    random_agent = RandomAgent()
    policy_iteration_agent = PolicyIterationAgent()
    value_iteration_agent = ValueIterationAgent()
    es_mc_agent = ESMonteCarloAgent()
    on_policy_mc_agent = OnPolicyMonteCarloAgent()
    normal_off_policy_mc_agent = \
        OffPolicyMonteCarloAgent(1.0, ImportanceSampling.NORMAL, 0.1, 1.0, 0.01)
    weighted_off_policy_mc_agent = \
        OffPolicyMonteCarloAgent(1.0, ImportanceSampling.WEIGHTED, 0.1, 1.0,
                                 0.01)
    ql_agent = QLearningAgent()
    sarsa_agent = SarsaAgent()
    expected_sarsa_agent = ExpectedSarsaAgent()
    double_ql_agent = DoubleQLearningAgent()
    double_sarsa_agent = DoubleSarsaAgent()
    double_expected_sarsa_agent = DoubleExpectedSarsaAgent()
    n_step_sarsa_agent = NStepSarsaAgent(0.5, 1.0, 2)
    n_step_expected_sarsa_agent = NStepExpectedSarsaAgent(0.5, 1.0, 2)
    off_policy_n_step_sarsa_agent = OffPolicyNStepSarsaAgent(0.5, 1.0, 2)
    off_policy_n_step_expected_sarsa_agent = \
        OffPolicyNStepExpectedSarsaAgent(0.5, 1.0, 2)
    n_step_tree_backup_agent = NStepTreeBackupAgent(0.5, 1.0, 2)

    print("Testing Policy Iteration...")
    game.set_first_player(policy_iteration_agent)
    game.set_second_player(optimal_agent)
    game.train()
    game.print_values()
    game.play(10000)

    print("Testing Value Iteration...")
    game.set_first_player(value_iteration_agent)
    game.set_second_player(optimal_agent)
    game.train()
    game.print_values()
    game.play(10000)

    print("Testing Exploring Start Monte Carlo...")
    game.set_first_player(es_mc_agent)
    game.set_second_player(optimal_agent)
    game.train(50000)
    game.print_values()
    game.play(10000)

    print("Testing On-policy Monte Carlo...")
    game.set_first_player(on_policy_mc_agent)
    game.set_second_player(on_policy_mc_agent)
    game.train(100000)
    game.print_values()
    game.set_second_player(optimal_agent)
    game.play(10000)

    print("Testing Off-policy Monte Carlo with Normal Sampling...")
    game.set_first_player(normal_off_policy_mc_agent)
    game.set_second_player(normal_off_policy_mc_agent)
    game.train(100000)
    game.print_values()
    game.set_second_player(optimal_agent)
    game.play(10000)

    print("Testing Off-policy Monte Carlo with Weighted Sampling...")
    game.set_first_player(weighted_off_policy_mc_agent)
    game.set_second_player(weighted_off_policy_mc_agent)
    game.train(100000)
    game.print_values()
    game.set_second_player(optimal_agent)
    game.play(10000)

    print("Testing Q Learning...")
    game.set_first_player(ql_agent)
    game.set_second_player(ql_agent)
    game.train(50000)
    game.print_values()
    game.set_second_player(optimal_agent)
    game.play(10000)

    print("Testing Sarsa...")
    game.set_first_player(sarsa_agent)
    game.set_second_player(sarsa_agent)
    game.train(50000)
    game.print_values()
    game.set_second_player(optimal_agent)
    game.play(10000)

    print("Testing Expected Sarsa...")
    game.set_first_player(expected_sarsa_agent)
    game.set_second_player(expected_sarsa_agent)
    game.train(50000)
    game.print_values()
    game.set_second_player(optimal_agent)
    game.play(10000)

    print("Testing Double Q Learning...")
    game.set_first_player(double_ql_agent)
    game.set_second_player(double_ql_agent)
    game.train(50000)
    game.print_values()
    game.set_second_player(optimal_agent)
    game.play(10000)

    print("Testing Double Sarsa...")
    game.set_first_player(double_sarsa_agent)
    game.set_second_player(double_sarsa_agent)
    game.train(50000)
    game.print_values()
    game.set_second_player(optimal_agent)
    game.play(10000)

    print("Testing Double Expected Sarsa...")
    game.set_first_player(double_expected_sarsa_agent)
    game.set_second_player(double_expected_sarsa_agent)
    game.train(50000)
    game.print_values()
    game.set_second_player(optimal_agent)
    game.play(10000)

    print("Testing n-step Sarsa...")
    game.set_first_player(n_step_sarsa_agent)
    game.set_second_player(n_step_sarsa_agent)
    game.train(50000)
    game.print_values()
    game.set_second_player(optimal_agent)
    game.play(10000)

    print("Testing n-step Expected Sarsa...")
    game.set_first_player(n_step_expected_sarsa_agent)
    game.set_second_player(n_step_expected_sarsa_agent)
    game.train(50000)
    game.print_values()
    game.set_second_player(optimal_agent)
    game.play(10000)

    print("Testing n-step Tree Backup...")
    game.set_first_player(n_step_tree_backup_agent)
    game.set_second_player(n_step_tree_backup_agent)
    game.train(50000)
    game.print_values()
    game.set_second_player(optimal_agent)
    game.play(10000)
