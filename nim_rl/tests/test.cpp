// Copyright 2020 Zhou Zikang. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "nim_rl/agent/dp_agent.h"
#include "nim_rl/agent/human_agent.h"
#include "nim_rl/agent/monte_carlo_agent.h"
#include "nim_rl/agent/n_step_bootstrapping_agent.h"
#include "nim_rl/agent/optimal_agent.h"
#include "nim_rl/agent/random_agent.h"
#include "nim_rl/agent/td_agent.h"
#include "nim_rl/environment/game.h"
#include "nim_rl/state/state.h"

using namespace nim_rl;

int main() {
  Game game(State({10, 10, 10}));
  HumanAgent human_agent;
  OptimalAgent optimal_agent;
  RandomAgent random_agent;
  PolicyIterationAgent policy_iteration_agent;
  ValueIterationAgent value_iteration_agent;
  ESMonteCarloAgent es_mc_agent;
  OnPolicyMonteCarloAgent on_policy_mc_agent;
  OffPolicyMonteCarloAgent normal_off_policy_mc_agent(
      1.0, ImportanceSampling::kNormal, 0.1, 1.0, 0.01);
  OffPolicyMonteCarloAgent weighted_off_policy_mc_agent(
      1.0, ImportanceSampling::kWeighted, 0.1, 1.0, 0.01);
  QLearningAgent ql_agent;
  SarsaAgent sarsa_agent;
  ExpectedSarsaAgent expected_sarsa_agent;
  DoubleQLearningAgent double_ql_agent;
  DoubleSarsaAgent double_sarsa_agent;
  DoubleExpectedSarsaAgent double_expected_sarsa_agent;
  NStepSarsaAgent n_step_sarsa_agent(0.5, 1.0, 2);
  NStepExpectedSarsaAgent n_step_expected_sarsa_agent(0.5, 1.0, 2);
  OffPolicyNStepSarsaAgent off_policy_n_step_sarsa_agent(0.5, 1.0, 2);
  OffPolicyNStepExpectedSarsaAgent off_policy_n_step_expected_sarsa_agent(
      0.5, 1.0, 2);
  NStepTreeBackupAgent n_step_tree_backup_agent(0.5, 1.0, 2);

  std::cout << "Testing Policy Iteration..." << std::endl;
  game.SetFirstPlayer(policy_iteration_agent);
  game.SetSecondPlayer(optimal_agent);
  game.Train();
  game.PrintValues();
  game.Play(10000);

  std::cout << "Testing Value Iteration..." << std::endl;
  game.SetFirstPlayer(value_iteration_agent);
  game.SetSecondPlayer(optimal_agent);
  game.Train();
  game.PrintValues();
  game.Play(10000);

  std::cout << "Testing Exploring Start Monte Carlo..." << std::endl;
  game.SetFirstPlayer(es_mc_agent);
  game.SetSecondPlayer(optimal_agent);
  game.Train(50000);
  game.PrintValues();
  game.SetSecondPlayer(optimal_agent);
  game.Play(10000);

  std::cout << "Testing On-policy Monte Carlo..." << std::endl;
  game.SetFirstPlayer(on_policy_mc_agent);
  game.SetSecondPlayer(on_policy_mc_agent);
  game.Train(100000);
  game.PrintValues();
  game.SetSecondPlayer(optimal_agent);
  game.Play(10000);

  std::cout << "Testing Off-policy Monte Carlo with Normal Sampling..."
            << std::endl;
  game.SetFirstPlayer(normal_off_policy_mc_agent);
  game.SetSecondPlayer(normal_off_policy_mc_agent);
  game.Train(100000);
  game.PrintValues();
  game.SetSecondPlayer(optimal_agent);
  game.Play(10000);

  std::cout << "Testing Off-policy Monte Carlo with Weighted Sampling..."
            << std::endl;
  game.SetFirstPlayer(weighted_off_policy_mc_agent);
  game.SetSecondPlayer(weighted_off_policy_mc_agent);
  game.Train(100000);
  game.PrintValues();
  game.SetSecondPlayer(optimal_agent);
  game.Play(10000);

  std::cout << "Testing Q Learning..." << std::endl;
  game.SetFirstPlayer(ql_agent);
  game.SetSecondPlayer(ql_agent);
  game.Train(50000);
  game.PrintValues();
  game.SetSecondPlayer(optimal_agent);
  game.Play(10000);

  std::cout << "Testing Sarsa..." << std::endl;
  game.SetFirstPlayer(sarsa_agent);
  game.SetSecondPlayer(sarsa_agent);
  game.Train(50000);
  game.PrintValues();
  game.SetSecondPlayer(optimal_agent);
  game.Play(10000);

  std::cout << "Testing Expected Sarsa..." << std::endl;
  game.SetFirstPlayer(expected_sarsa_agent);
  game.SetSecondPlayer(expected_sarsa_agent);
  game.Train(50000);
  game.PrintValues();
  game.SetSecondPlayer(optimal_agent);
  game.Play(10000);

  std::cout << "Testing Double Q Learning..." << std::endl;
  game.SetFirstPlayer(double_ql_agent);
  game.SetSecondPlayer(double_ql_agent);
  game.Train(50000);
  game.PrintValues();
  game.SetSecondPlayer(optimal_agent);
  game.Play(10000);

  std::cout << "Testing Double Sarsa..." << std::endl;
  game.SetFirstPlayer(double_sarsa_agent);
  game.SetSecondPlayer(double_sarsa_agent);
  game.Train(50000);
  game.PrintValues();
  game.SetSecondPlayer(optimal_agent);
  game.Play(10000);

  std::cout << "Testing Double Expected Sarsa..." << std::endl;
  game.SetFirstPlayer(double_expected_sarsa_agent);
  game.SetSecondPlayer(double_expected_sarsa_agent);
  game.Train(50000);
  game.PrintValues();
  game.SetSecondPlayer(optimal_agent);
  game.Play(10000);

  std::cout << "Testing n-step Sarsa..." << std::endl;
  game.SetFirstPlayer(n_step_sarsa_agent);
  game.SetSecondPlayer(n_step_sarsa_agent);
  game.Train(50000);
  game.PrintValues();
  game.SetSecondPlayer(optimal_agent);
  game.Play(10000);

  std::cout << "Testing n-step Expected Sarsa..." << std::endl;
  game.SetFirstPlayer(n_step_expected_sarsa_agent);
  game.SetSecondPlayer(n_step_expected_sarsa_agent);
  game.Train(50000);
  game.PrintValues();
  game.SetSecondPlayer(optimal_agent);
  game.Play(10000);

  std::cout << "Testing Off-policy n-step Sarsa..." << std::endl;
  game.SetFirstPlayer(off_policy_n_step_sarsa_agent);
  game.SetSecondPlayer(off_policy_n_step_sarsa_agent);
  game.Train(50000);
  game.PrintValues();
  game.SetSecondPlayer(optimal_agent);
  game.Play(10000);

  std::cout << "Testing Off-policy n-step Expected Sarsa..." << std::endl;
  game.SetFirstPlayer(off_policy_n_step_expected_sarsa_agent);
  game.SetSecondPlayer(off_policy_n_step_expected_sarsa_agent);
  game.Train(50000);
  game.PrintValues();
  game.SetSecondPlayer(optimal_agent);
  game.Play(10000);

  std::cout << "Testing n-step Tree Backup..." << std::endl;
  game.SetFirstPlayer(n_step_tree_backup_agent);
  game.SetSecondPlayer(n_step_tree_backup_agent);
  game.Train(50000);
  game.PrintValues();
  game.SetSecondPlayer(optimal_agent);
  game.Play(10000);

  return 0;
}
