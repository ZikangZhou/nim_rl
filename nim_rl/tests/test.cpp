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

void ValueFunctionsTest() {
  for (int n = 1; n <= 10; ++n) {
    State initial_state(3, n);

    std::vector<State> all_states;
    all_states.reserve(initial_state[0] + 1);
    for (int num_objects = 0; num_objects != initial_state[0] + 1;
         ++num_objects)
      all_states.emplace_back(1, num_objects);
    std::vector<State> new_all_states;
    for (int pile_id = 1; pile_id < initial_state.Size(); ++pile_id) {
      for (const auto &state : all_states) {
        for (unsigned num_objects = 0;
             num_objects != initial_state[pile_id] + 1; ++num_objects) {
          State next_state(state.Size() + 1, 0);
          for (int i = 0; i < state.Size(); ++i)
            next_state[i] = state[i];
          next_state[state.Size()] = num_objects;
          new_all_states.emplace_back(std::move(next_state));
        }
      }
      std::swap(all_states, new_all_states);
      new_all_states.clear();
    }

    int action_items = 0, after_state_items = all_states.size();
    for (const auto &state : all_states) {
      action_items += state.LegalActions().size();
    }
    std::cout << action_items << " " << after_state_items << " "
              << static_cast<double>(action_items - after_state_items)
                  / action_items << std::endl;
  }
}

void SortOrNotTest() {
  for (int n = 1; n <= 10; ++n) {
    State initial_state(3, n);

    std::vector<State> all_states;
    all_states.reserve(initial_state[0] + 1);
    for (int num_objects = 0; num_objects != initial_state[0] + 1;
         ++num_objects)
      all_states.emplace_back(1, num_objects);
    std::vector<State> new_all_states;
    for (int pile_id = 1; pile_id < initial_state.Size(); ++pile_id) {
      for (const auto &state : all_states) {
        for (unsigned num_objects = 0;
             num_objects != initial_state[pile_id] + 1; ++num_objects) {
          State next_state(state.Size() + 1, 0);
          for (int i = 0; i < state.Size(); ++i)
            next_state[i] = state[i];
          next_state[state.Size()] = num_objects;
          new_all_states.emplace_back(std::move(next_state));
        }
      }
      std::swap(all_states, new_all_states);
      new_all_states.clear();
    }

    int sort_items = initial_state.GetAllStates().size(),
        not_sort_items = all_states.size();
    std::cout << sort_items << " " << not_sort_items << " "
              << static_cast<double>(not_sort_items - sort_items)
                  / not_sort_items
              << std::endl;
  }
}

void RandomExperiment(Agent *first_player, Agent *second_player, int round) {
  Game game(State({10, 10, 10}));
  std::vector<double> average_optimal_ratios(50000 / kCheckPoint + 1, 0.0);
  std::vector<double> average_mean_square_errors(500000 / kCheckPoint + 1, 0.0);
  for (int i = 1; i <= round; ++i) {
    game.SetFirstPlayer(*first_player);
    game.SetSecondPlayer(*second_player);
    auto monitor_data = game.Train(50000);
    std::vector<double> optimal_ratios = monitor_data.first;
    std::vector<double> mean_square_errors = monitor_data.second;
    for (int ckpt = 0; ckpt < average_optimal_ratios.size(); ++ckpt) {
      average_optimal_ratios[ckpt] += optimal_ratios[ckpt];
    }
    for (int ckpt = 0; ckpt < average_mean_square_errors.size(); ++ckpt) {
      average_mean_square_errors[ckpt] += mean_square_errors[ckpt];
    }
  }
  std::cout << std::fixed << std::setprecision(kPrecision);
  for (auto &average_optimal_ratio : average_optimal_ratios) {
    average_optimal_ratio /= round;
    std::cout << average_optimal_ratio << std::endl;
  }
  for (auto &average_mean_square_error : average_mean_square_errors) {
    average_mean_square_error /= round;
    std::cout << average_mean_square_error << std::endl;
  }
}

void MonteCarloTest() {
  Game game(State({10, 10, 10}));
  OnPolicyMonteCarloAgent on_policy_mc_agent(1.0, EpsilonGreedy(0.1, 1.0, 0.1));
  OffPolicyMonteCarloAgent normal_off_policy_mc_agent(
      1.0, ImportanceSampling::kNormal, 0.1, 1.0, 0.1);
  OffPolicyMonteCarloAgent weighted_off_policy_mc_agent(
      1.0, ImportanceSampling::kWeighted, 0.1, 1.0, 0.1);
  NStepSarsaAgent n_step_sarsa_agent(0.5, 1.0, 20);
  OffPolicyNStepSarsaAgent off_policy_n_step_sarsa_agent(0.5, 1.0, 20);

  std::cout << "Testing On-policy Monte Carlo..." << std::endl;
  RandomExperiment(&on_policy_mc_agent, &on_policy_mc_agent, 10);

  std::cout << "Testing Off-policy Monte Carlo with Normal Sampling..."
            << std::endl;
  RandomExperiment(&normal_off_policy_mc_agent,
                   &normal_off_policy_mc_agent,
                   10);

  std::cout << "Testing Off-policy Monte Carlo with Weighted Sampling..."
            << std::endl;
  RandomExperiment(&weighted_off_policy_mc_agent,
                   &weighted_off_policy_mc_agent,
                   10);

  std::cout << "Testing alpha MC..." << std::endl;
  RandomExperiment(&n_step_sarsa_agent,
                   &n_step_sarsa_agent,
                   10);

  std::cout << "Testing Off-policy alpha MC..." << std::endl;
  RandomExperiment(&off_policy_n_step_sarsa_agent,
                   &off_policy_n_step_sarsa_agent,
                   10);
}

void OpponentTest() {
  Game game(State({10, 10, 10}));
  QLearningAgent ql_agent;
  OptimalAgent optimal_agent;
  RandomAgent random_agent;

  std::cout << "Testing Q Learning agent vs random agent..." << std::endl;
  RandomExperiment(&ql_agent, &random_agent, 10);

  std::cout << "Testing Q Learning agent vs optimal agent..." << std::endl;
  RandomExperiment(&ql_agent, &optimal_agent, 10);

  std::cout << "Testing Q Learning agent vs Q Learning agent..." << std::endl;
  RandomExperiment(&ql_agent, new QLearningAgent, 10);

  std::cout << "Testing Q Learningagent vs self..." << std::endl;
  RandomExperiment(&ql_agent, &ql_agent, 10);
}

void TDTest() {
  Game game(State({10, 10, 10}));
  QLearningAgent ql_agent;
  SarsaAgent sarsa_agent;
  ExpectedSarsaAgent expected_sarsa_agent;
  DoubleQLearningAgent double_ql_agent;
  DoubleSarsaAgent double_sarsa_agent;
  DoubleExpectedSarsaAgent double_expected_sarsa_agent;

  std::cout << "Testing Q Learning..." << std::endl;
  RandomExperiment(&ql_agent, &ql_agent, 10);

  std::cout << "Testing Sarsa..." << std::endl;
  RandomExperiment(&sarsa_agent, &sarsa_agent, 10);

  std::cout << "Testing Expected Sarsa..." << std::endl;
  RandomExperiment(&expected_sarsa_agent, &expected_sarsa_agent, 10);

  std::cout << "Testing Double Q Learning..." << std::endl;
  RandomExperiment(&double_ql_agent, &double_ql_agent, 10);

  std::cout << "Testing Double Sarsa..." << std::endl;
  RandomExperiment(&double_sarsa_agent, &double_sarsa_agent, 10);

  std::cout << "Testing Double Expected Sarsa..." << std::endl;
  RandomExperiment(&double_expected_sarsa_agent,
                   &double_expected_sarsa_agent,
                   10);
}

void NStepTest() {
  Game game(State({10, 10, 10}));
  for (int n = 6; n <= 7; n += 1) {
    NStepSarsaAgent n_step_sarsa_agent(0.5, 1.0, n);
    std::cout << "Testing " << n << "-step Sarsa..." << std::endl;
    RandomExperiment(&n_step_sarsa_agent, &n_step_sarsa_agent, 10);
  }
  for (int n = 6; n <= 7; n += 1) {
    NStepExpectedSarsaAgent n_step_expected_sarsa_agent(0.5, 1.0, n);
    std::cout << "Testing " << n << "-step Expected Sarsa..." << std::endl;
    RandomExperiment(&n_step_expected_sarsa_agent,
                     &n_step_expected_sarsa_agent,
                     10);
  }
  for (int n = 6; n <= 7; n += 1) {
    OffPolicyNStepSarsaAgent off_policy_n_step_sarsa_agent(0.5, 1.0, n);
    std::cout << "Testing Off-policy " << n << "-step Sarsa..." << std::endl;
    RandomExperiment(&off_policy_n_step_sarsa_agent,
                     &off_policy_n_step_sarsa_agent,
                     10);
  }
  for (int n = 6; n <= 7; n += 1) {
    OffPolicyNStepExpectedSarsaAgent
        off_policy_n_step_expected_sarsa_agent(0.5, 1.0, n);
    std::cout << "Testing Off-policy " << n << "-step Expected Sarsa..."
              << std::endl;
    RandomExperiment(&off_policy_n_step_expected_sarsa_agent,
                     &off_policy_n_step_expected_sarsa_agent,
                     10);
  }
  for (int n = 6; n <= 7; n += 1) {
    NStepTreeBackupAgent n_step_tree_backup_agent(0.5, 1.0, n);
    std::cout << "Testing " << n << "-step Tree Backup..." << std::endl;
    RandomExperiment(&n_step_tree_backup_agent,
                     &n_step_tree_backup_agent,
                     10);
  }
}

void ValueFunctionSizesTest() {
  for (int num_pile = 1; num_pile <= 10; ++num_pile) {
    Game game(State(num_pile, 10));
    std::cout << game.GetAllStates().size() << std::endl;
  }
}

void AverageEpisodeSize() {
  Game game(State({10, 10, 10}));
  RandomAgent optimal_agent;
  game.SetFirstPlayer(optimal_agent);
  game.SetSecondPlayer(optimal_agent);
  game.Play(10000000);
}

void StateSpaceTest() {
  for (int pile = 6; pile <= 10; ++pile) {
    State state = State(pile, 10);
    Game game(state);
    QLearningAgent ql_agent;
    std::cout << "Testing Q Learning with state: " << state << std::endl;
    game.SetFirstPlayer(ql_agent);
    game.SetSecondPlayer(ql_agent);
    game.Train(500000);
  }
}

int main() {
  Game game(State({5, 5, 5}));
  HumanAgent human_agent;
  OptimalAgent optimal_agent;
  RandomAgent random_agent;
  PolicyIterationAgent policy_iteration_agent;
  ValueIterationAgent value_iteration_agent;
  ESMonteCarloAgent es_mc_agent;
  OnPolicyMonteCarloAgent on_policy_mc_agent(1.0,
                                             EpsilonGreedy(0.1, 1.0, 0.1));
  OffPolicyMonteCarloAgent normal_off_policy_mc_agent(
      1.0, ImportanceSampling::kNormal, 0.1, 1.0, 0.1);
  OffPolicyMonteCarloAgent weighted_off_policy_mc_agent(
      1.0, ImportanceSampling::kWeighted, 0.1, 1.0, 0.1);
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
  game.SetSecondPlayer(es_mc_agent);
  game.Train(50000);
  game.PrintValues();
  game.SetSecondPlayer(optimal_agent);
  game.Play(10000);

  std::cout << "Testing On-policy Monte Carlo..." << std::endl;
  game.SetFirstPlayer(on_policy_mc_agent);
  game.SetSecondPlayer(on_policy_mc_agent);
  game.Train(50000);
  game.PrintValues();
  game.SetSecondPlayer(optimal_agent);
  game.Play(10000);

  std::cout << "Testing Off-policy Monte Carlo with Normal Sampling..."
            << std::endl;
  game.SetFirstPlayer(normal_off_policy_mc_agent);
  game.SetSecondPlayer(normal_off_policy_mc_agent);
  game.Train(50000);
  game.PrintValues();
  game.SetSecondPlayer(optimal_agent);
  game.Play(10000);

  std::cout << "Testing Off-policy Monte Carlo with Weighted Sampling..."
            << std::endl;
  game.SetFirstPlayer(weighted_off_policy_mc_agent);
  game.SetSecondPlayer(weighted_off_policy_mc_agent);
  game.Train(50000);
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
