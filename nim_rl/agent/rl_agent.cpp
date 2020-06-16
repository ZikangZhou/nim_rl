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

#include "nim_rl/agent/rl_agent.h"
#include "nim_rl/environment/game.h"

namespace nim_rl {

void RLAgent::Initialize(const std::vector<State> &all_states) {
  for (const auto &state : all_states) {
    if (state.IsTerminal()) {
      (*values_)[state] = kWinReward;
    } else {
      (*values_)[state] = kTieReward;
    }
  }
  (*values_)[State()] = kTieReward;
}

double RLAgent::MinSquareError() {
  int cnt = 0;
  double error = 0.0;
  Values values = GetValues();
  for (const auto &kv : values) {
    if (kv.first.IsEmpty()) continue;
    ++cnt;
    if (kv.first.NimSum()) {
      error += (kv.second - kLoseReward) * (kv.second - kLoseReward);
    } else {
      error += (kv.second - kWinReward) * (kv.second - kWinReward);
    }
  }
  return error / cnt;
}

double RLAgent::OptimalActionsRatio() {
  double num_n_positions = 0.0;
  double num_optimal_actions = 0.0;
  Values values = GetValues();
  std::cout << values << std::endl;
  for (const auto &kv : values) {
    if (kv.first.NimSum()) {
      ++num_n_positions;
      std::vector<Action> legal_actions = kv.first.LegalActions();
      Action greedy_action =
          *std::max_element(legal_actions.begin(), legal_actions.end(),
                            [&](const Action &a1, const Action &a2) {
                              return values[kv.first.Child(a1)]
                                  < values[kv.first.Child(a2)];
                            });
      if (!kv.first.Child(greedy_action).NimSum())
        ++num_optimal_actions;
    }
  }
  return num_optimal_actions / num_n_positions;
}

Action RLAgent::Policy(const State &state, bool is_evaluation) {
  legal_actions_ = state.LegalActions();
  greedy_actions_.clear();
  if (legal_actions_.empty()) {
    greedy_value_ = 0.0;
    return Action{};
  } else {
    Action greedy_action =
        *std::max_element(legal_actions_.begin(), legal_actions_.end(),
                          [&](const Action &a1, const Action &a2) {
                            return (*values_)[state.Child(a1)]
                                < (*values_)[state.Child(a2)];
                          });
    greedy_value_ = (*values_)[state.Child(greedy_action)];
    std::copy_if(legal_actions_.begin(), legal_actions_.end(),
                 std::back_inserter(greedy_actions_),
                 [&](const Action &action) {
                   return (*values_)[state.Child(action)] == greedy_value_;
                 });
    if (is_evaluation) {
      return SampleAction(greedy_actions_);
    } else {
      return PolicyImpl(legal_actions_, greedy_actions_);
    }
  }
}

void RLAgent::Reset() {
  Agent::Reset();
  greedy_value_ = 0.0;
  legal_actions_.clear();
  greedy_actions_.clear();
}

std::ostream &operator<<(std::ostream &os,
                         const std::unordered_map<State,
                                                  Agent::Reward> &values) {
  os << std::fixed << std::setprecision(kPrecision);
  for (const auto &value : values)
    os << value.first << ": " << value.second << " ";
  return os;
}

std::ostream &operator<<(std::ostream &os,
                         const std::vector<RLAgent::TimeStep> &trajectory) {
  for (const auto &time_step : trajectory)
    os << std::get<0>(time_step) << ", " << std::get<1>(time_step) << ", "
       << std::get<2>(time_step) << "; ";
  return os;
}

}  // namespace nim_rl
