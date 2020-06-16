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

#include "nim_rl/agent/monte_carlo_agent.h"
#include "nim_rl/environment/game.h"

namespace nim_rl {

void MonteCarloAgent::Initialize(const std::vector<State> &all_states) {
  RLAgent::Initialize(all_states);
  for (const auto &kv : *values_) {
    cumulative_sums_[kv.first] = 0.0;
  }
}

void MonteCarloAgent::Reset() {
  RLAgent::Reset();
  trajectory_.clear();
}

Action MonteCarloAgent::Step(Game *game, bool is_evaluation) {
  if (!trajectory_.empty())
    std::get<2>(trajectory_.back()) = -game->GetReward();
  State state = game->GetState();
  Action action = Agent::Step(game, is_evaluation);
  trajectory_.emplace_back(state, action, game->GetReward());
  if (game->GetState().IsTerminal() || game->GetState().IsEmpty())
    Update(State(), State(), 0);
  return action;
}

void MonteCarloAgent::Update(const State &/*update_state*/,
                             const State &/*current_state*/,
                             Reward /*reward*/) {
  double ret = 0.0;
  for (auto r_iter = trajectory_.crbegin(); r_iter != trajectory_.crend();
       ++r_iter) {
    const State &state = std::get<0>(*r_iter);
    if (!state.IsTerminal()) {
      const Action &action = std::get<1>(*r_iter);
      const State &next_state = state.Child(action);
      ret = gamma_ * ret + std::get<2>(*r_iter);
      if (std::find_if(r_iter + 1, trajectory_.crend(),
                       [&](const TimeStep &time_step) {
                         return std::get<0>(time_step).Child(action) ==
                             next_state;
                       }) == trajectory_.crend()) {
        ++cumulative_sums_[next_state];
        (*values_)[next_state] +=
            (ret - (*values_)[next_state]) / cumulative_sums_[next_state];
      }
    }
  }
}

Action ESMonteCarloAgent::Step(Game *game, bool is_evaluation) {
  if (!is_evaluation && game->GetState() == game->GetInitialState()) {
    State start_state = SampleState(game->GetAllStates());
    Action start_action = SampleAction(start_state.LegalActions());
    game->SetState(start_state);
    game->Step(start_action);
    trajectory_.emplace_back(start_state, start_action, game->GetReward());
    return start_action;
  } else {
    return MonteCarloAgent::Step(game, is_evaluation);
  }
}

void OffPolicyMonteCarloAgent::Update(const State &/*update_state*/,
                                      const State &/*current_state*/,
                                      Reward /*reward*/) {
  double ret = 0.0, weight = 1.0;
  double epsilon = epsilon_greedy_.GetEpsilon();
  std::shared_ptr<Values> behavior_policy_values =
      std::make_shared<Values>(*values_);
  for (auto r_iter = trajectory_.crbegin(); r_iter != trajectory_.crend();
       ++r_iter) {
    const State &state = std::get<0>(*r_iter);
    if (state.IsTerminal()) continue;
    const Action &action = std::get<1>(*r_iter);
    const State &next_state = state.Child(action);
    ret = gamma_ * ret + std::get<2>(*r_iter);
    if (importance_sampling_ == ImportanceSampling::kNormal) {
      ++cumulative_sums_[next_state];
      (*values_)[next_state] +=
          (weight * ret - (*values_)[next_state])
              / cumulative_sums_[next_state];
    } else if (importance_sampling_ == ImportanceSampling::kWeighted) {
      cumulative_sums_[next_state] += weight;
      (*values_)[next_state] +=
          weight * (ret - (*values_)[next_state])
              / cumulative_sums_[next_state];
    }
    std::vector<Action> legal_actions = state.LegalActions();
    int num_legal_actions = legal_actions.size();
    Action target_policy_greedy_action =
        *std::max_element(legal_actions.begin(), legal_actions.end(),
                          [&](const Action &a1, const Action &a2) {
                            return (*values_)[state.Child(a1)]
                                < (*values_)[state.Child(a2)];
                          });
    double target_policy_greedy_value =
        (*values_)[state.Child(target_policy_greedy_action)];
    if ((*values_)[next_state] != target_policy_greedy_value) break;
    int num_target_policy_greedy_actions =
        std::count_if(legal_actions.begin(), legal_actions.end(),
                      [&](const Action &action) {
                        return (*values_)[state.Child(action)] ==
                            target_policy_greedy_value;
                      });
    double behavior_policy_value = (*behavior_policy_values)[next_state];
    Action behavior_policy_greedy_action =
        *std::max_element(legal_actions.begin(), legal_actions.end(),
                          [&](const Action &a1, const Action &a2) {
                            return (*behavior_policy_values)[state.Child(a1)] <
                                (*behavior_policy_values)[state.Child(a2)];
                          });
    double behavior_policy_greedy_value =
        (*behavior_policy_values)[state.Child(behavior_policy_greedy_action)];
    int num_behavior_policy_greedy_actions =
        std::count_if(legal_actions.begin(), legal_actions.end(),
                      [&](const Action &action) {
                        return (*behavior_policy_values)[state.Child(action)] ==
                            behavior_policy_greedy_value;
                      });
    if (behavior_policy_value == behavior_policy_greedy_value) {
      weight *= ((1 - epsilon) / num_target_policy_greedy_actions
          + epsilon / num_legal_actions)
          / ((1 - epsilon) / num_behavior_policy_greedy_actions
              + epsilon / num_legal_actions);
    } else {
      weight *= ((1 - epsilon) / num_target_policy_greedy_actions
          + epsilon / num_legal_actions)
          / (epsilon / num_legal_actions);
    }
  }
}

}  // namespace nim_rl
