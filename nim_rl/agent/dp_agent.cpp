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
#include "nim_rl/agent/optimal_agent.h"
#include "nim_rl/environment/game.h"

namespace nim_rl {

void DPAgent::Initialize(const std::vector<State> &all_states) {
  OptimalAgent optimal_agent;
  for (const auto &state : all_states) {
    if (state.IsTerminal()) {
      values_.insert({state, kWinReward});
    } else {
      values_.insert({state, kTieReward});
      std::vector<Action> legal_actions = state.LegalActions();
      legal_actions.emplace_back();
      for (const auto &action : legal_actions) {
        State next_state = state.Child(action);
        std::vector<StateProb> possibilities;
        if (next_state.NimSum()) {
          possibilities.emplace_back(
              next_state.Child(optimal_agent.Policy(next_state, true)), 1.0);
        } else {
          std::vector<Action> next_legal_actions = next_state.LegalActions();
          double prob = 1.0 / next_legal_actions.size();
          for (const auto &next_legal_action : next_legal_actions)
            possibilities.emplace_back(next_state.Child(next_legal_action),
                                       prob);
        }
        transitions_.insert({{state, action}, possibilities});
      }
    }
  }
}

void PolicyIterationAgent::Initialize(const std::vector<State> &all_states) {
  DPAgent::Initialize(all_states);
  for (const auto &state : all_states)
    policy_.insert({state, SampleAction(state.LegalActions())});
  PolicyIteration(all_states);
}

void PolicyIterationAgent::PolicyIteration(const
                                           std::vector<State> &all_states) {
  int step = 0;
  bool policy_stable = false;
  while (!policy_stable) {
    double delta;
    do {
      delta = 0.0;
      for (const auto &state : all_states) {
        if (state.IsTerminal()) continue;
        Value value = 0.0;
        auto possibilities = transitions_[{state, Action()}];
        for (const auto &outcome : possibilities) {
          Reward reward = outcome.first.IsTerminal() ? kLoseReward : kTieReward;
          value += outcome.second * (reward + gamma_
              * values_[outcome.first.Child(Policy(outcome.first, false))]);
        }
        Value *stored_value = &values_[state];
        delta = std::max(delta, std::abs(*stored_value - value));
        *stored_value = value;
      }
    } while (delta > threshold_);
    policy_stable = true;
    for (const auto &state : all_states) {
      if (state.IsTerminal()) continue;
      Action old_action = policy_[state];
      std::vector<Action> legal_actions = state.LegalActions();
      policy_[state] =
          *std::max_element(legal_actions.begin(), legal_actions.end(),
                            [&](const Action &a1, const Action &a2) {
                              return values_[state.Child(a1)]
                                  < values_[state.Child(a2)];
                            });
      if (old_action != policy_[state]) policy_stable = false;
    }
    std::cout << std::fixed << std::setprecision(kPrecision) << "Epoch "
              << ++step << ": Policy Iteration agent optimal actions ratio: "
              << OptimalActionsRatio() << std::endl;
  }
}

void ValueIterationAgent::Initialize(const std::vector<State> &all_states) {
  DPAgent::Initialize(all_states);
  ValueIteration(all_states);
}

void ValueIterationAgent::ValueIteration(const std::vector<State> &all_states) {
  int step = 0;
  double delta;
  do {
    delta = 0.0;
    for (const auto &state : all_states) {
      if (state.IsTerminal()) continue;
      Value value = 0.0;
      auto possibilities = transitions_[{state, Action()}];
      for (const auto &outcome : possibilities) {
        Reward reward = outcome.first.IsTerminal() ? kLoseReward : kTieReward;
        value += outcome.second * (reward + gamma_
            * values_[outcome.first.Child(Policy(outcome.first, false))]);
      }
      Value *stored_value = &values_[state];
      delta = std::max(delta, std::abs(*stored_value - value));
      *stored_value = value;
    }
    std::cout << std::fixed << std::setprecision(kPrecision) << "Epoch "
              << ++step << ": Value Iteration agent optimal actions ratio: "
              << OptimalActionsRatio() << std::endl;
  } while (delta > threshold_);
}

}  // namespace nim_rl
