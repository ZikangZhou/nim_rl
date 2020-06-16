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

#include "nim_rl/agent/n_step_bootstrapping_agent.h"
#include "nim_rl/environment/game.h"

namespace nim_rl {

void NStepBootstrappingAgent::Reset() {
  TDAgent::Reset();
  current_time_ = 0;
  terminal_time_ = INT_MAX;
  update_time_ = 0;
  trajectory_.clear();
}

Action NStepBootstrappingAgent::Step(Game *game, bool is_evaluation) {
  Action action;
  if (!trajectory_.empty())
    std::get<2>(trajectory_.back()) = -game->GetReward();
  State state = game->GetState();
  if (state.IsTerminal()) std::get<2>(trajectory_.back()) = 0.0;
  action = Policy(state, is_evaluation);
  game->Step(action);
  trajectory_.emplace_back(state, action, game->GetReward());
  if (game->GetState().IsTerminal() || game->GetState().IsEmpty())
    terminal_time_ = current_time_ + 1;
  if (!is_evaluation) {
    update_time_ = current_time_ - n_;
    if (update_time_ >= 0) {
      TimeStep time_step = trajectory_[update_time_];
      State update_state =
          std::get<0>(time_step).Child(std::get<1>(time_step));
      Update(update_state, game->GetState(), 0.0);
    }
    if (game->GetState().IsTerminal() || game->GetState().IsEmpty()) {
      while (++update_time_ < terminal_time_ - 1) {
        if (update_time_ >= 0) {
          TimeStep time_step = trajectory_[update_time_];
          State update_state =
              std::get<0>(time_step).Child(std::get<1>(time_step));
          Update(update_state, game->GetState(), 0.0);
        }
      }
    }
  }
  ++current_time_;
  return action;
}

void NStepSarsaAgent::Update(const State &update_state,
                             const State &current_state,
                             Reward /*reward*/) {
  double ret = 0.0;
  for (int i = update_time_;
       i < std::min(update_time_ + n_ + 1, terminal_time_); ++i)
    ret += pow(gamma_, i - update_time_) * std::get<2>(trajectory_[i]);
  if (update_time_ + n_ < terminal_time_ - 1)
    ret += pow(gamma_, n_) * (*values_)[current_state];
  (*values_)[update_state] += alpha_ * (ret - (*values_)[update_state]);
}

Action NStepExpectedSarsaAgent::Policy(const State &state, bool is_evaluation) {
  SetNextStates(state.Children());
  return NStepBootstrappingAgent::Policy(state, is_evaluation);
}

void NStepExpectedSarsaAgent::Reset() {
  NStepBootstrappingAgent::Reset();
  next_states_.clear();
}

void NStepExpectedSarsaAgent::Update(const State &update_state,
                                     const State &/*current_state*/,
                                     Reward /*reward*/) {
  double ret = 0.0;
  for (int i = update_time_;
       i < std::min(update_time_ + n_ + 1, terminal_time_); ++i) {
    ret += pow(gamma_, i - update_time_) * std::get<2>(trajectory_[i]);
  }
  if (update_time_ + n_ < terminal_time_ - 1) {
    double expectation = 0.0;
    if (!legal_actions_.empty()) {
      double epsilon = epsilon_greedy_.GetEpsilon();
      for (const auto &state : next_states_)
        if ((*values_)[state] != greedy_value_)
          expectation += epsilon * (*values_)[state] / legal_actions_.size();
      expectation += (1 - epsilon) * greedy_value_
          + greedy_actions_.size() * epsilon * greedy_value_
              / legal_actions_.size();
    }
    ret += pow(gamma_, n_) * expectation;
  }
  (*values_)[update_state] += alpha_ * (ret - (*values_)[update_state]);
}

void OffPolicyNStepSarsaAgent::Update(const State &update_state,
                                      const State &current_state,
                                      Reward /*reward*/) {
  double ret = 0.0, weight = 1.0;
  double epsilon = epsilon_greedy_.GetEpsilon();
  for (int i = update_time_ + 1;
       i < std::min(update_time_ + n_ + 1, terminal_time_); ++i) {
    const State &state = std::get<0>(trajectory_[i]);
    if (!state.IsTerminal()) {
      const Action &action = std::get<1>(trajectory_[i]);
      const State &next_state = state.Child(action);
      double value = (*values_)[next_state];
      std::vector<Action> legal_actions = state.LegalActions();
      Action greedy_action =
          *std::max_element(legal_actions.begin(), legal_actions.end(),
                            [&](const Action &a1, const Action &a2) {
                              return (*values_)[state.Child(a1)] <
                                  (*values_)[state.Child(a2)];
                            });
      double greedy_value = (*values_)[state.Child(greedy_action)];
      if (value != greedy_value) {
        weight = 0.0;
        break;
      } else {
        int num_legal_actions = legal_actions.size();
        int num_greedy_actions =
            std::count_if(legal_actions.begin(), legal_actions.end(),
                          [&](const Action &action) {
                            return (*values_)[state.Child(action)] ==
                                greedy_value;
                          });
        weight *= num_legal_actions / ((1 - epsilon) * num_legal_actions
            + epsilon * num_greedy_actions);
      }
    }
  }
  for (int i = update_time_;
       i < std::min(update_time_ + n_ + 1, terminal_time_); ++i) {
    ret += pow(gamma_, i - update_time_) * std::get<2>(trajectory_[i]);
  }
  if (update_time_ + n_ < terminal_time_ - 1) {
    ret += pow(gamma_, n_) * (*values_)[current_state];
  }
  (*values_)[update_state] +=
      alpha_ * (weight * ret - (*values_)[update_state]);
}

Action OffPolicyNStepExpectedSarsaAgent::Policy(const State &state,
                                                bool is_evaluation) {
  SetNextStates(state.Children());
  return NStepBootstrappingAgent::Policy(state, is_evaluation);
}

void OffPolicyNStepExpectedSarsaAgent::Reset() {
  NStepBootstrappingAgent::Reset();
  next_states_.clear();
}

void OffPolicyNStepExpectedSarsaAgent::Update(const State &update_state,
                                              const State &/*current_state*/,
                                              Reward /*reward*/) {
  double ret = 0.0, weight = 1.0;
  double epsilon = epsilon_greedy_.GetEpsilon();
  for (int i = update_time_ + 1;
       i < std::min(update_time_ + n_ + 1, terminal_time_); ++i) {
    const State &state = std::get<0>(trajectory_[i]);
    if (!state.IsTerminal()) {
      const Action &action = std::get<1>(trajectory_[i]);
      const State &next_state = state.Child(action);
      double value = (*values_)[next_state];
      std::vector<Action> legal_actions = state.LegalActions();
      Action greedy_action =
          *std::max_element(legal_actions.begin(), legal_actions.end(),
                            [&](const Action &a1, const Action &a2) {
                              return (*values_)[state.Child(a1)] <
                                  (*values_)[state.Child(a2)];
                            });
      double greedy_value = (*values_)[state.Child(greedy_action)];
      if (value != greedy_value) {
        weight = 0.0;
        break;
      } else {
        int num_legal_actions = legal_actions.size();
        int num_greedy_actions =
            std::count_if(legal_actions.begin(), legal_actions.end(),
                          [&](const Action &action) {
                            return (*values_)[state.Child(action)] ==
                                greedy_value;
                          });
        weight *= num_legal_actions / ((1 - epsilon) * num_legal_actions
            + epsilon * num_greedy_actions);
      }
    }
  }
  for (int i = update_time_;
       i < std::min(update_time_ + n_ + 1, terminal_time_); ++i)
    ret += pow(gamma_, i - update_time_) * std::get<2>(trajectory_[i]);
  if (update_time_ + n_ < terminal_time_ - 1) {
    double expectation = 0.0;
    if (!legal_actions_.empty()) {
      for (const auto &state : next_states_)
        if ((*values_)[state] != greedy_value_)
          expectation += epsilon * (*values_)[state] / legal_actions_.size();
      expectation += (1 - epsilon) * greedy_value_
          + greedy_actions_.size() * epsilon * greedy_value_
              / legal_actions_.size();
    }
    ret += pow(gamma_, n_) * expectation;
  }
  (*values_)[update_state] +=
      alpha_ * (weight * ret - (*values_)[update_state]);
}

void NStepTreeBackupAgent::Update(const State &update_state,
                                  const State &/*current_state*/,
                                  Reward /*reward*/) {
  int backup_time = std::min(update_time_ + n_, terminal_time_ - 1);
  double ret = std::get<2>(trajectory_[backup_time]);
  State backup_state = std::get<0>(trajectory_[backup_time]);
  if (!backup_state.IsTerminal()) {
    std::vector<Action> legal_actions = backup_state.LegalActions();
    Action greedy_action =
        *std::max_element(legal_actions.begin(), legal_actions.end(),
                          [&](const Action &a1, const Action &a2) {
                            return (*values_)[backup_state.Child(a1)] <
                                (*values_)[backup_state.Child(a2)];
                          });
    Reward greedy_value = (*values_)[backup_state.Child(greedy_action)];
    ret = gamma_ * greedy_value;
  }
  for (int i = backup_time - 1; i > update_time_; --i) {
    State state_i = std::get<0>(trajectory_[i]);
    Action action_i = std::get<1>(trajectory_[i]);
    Reward reward_i = std::get<2>(trajectory_[i]);
    std::vector<Action> legal_actions = state_i.LegalActions();
    Action greedy_action =
        *std::max_element(legal_actions.begin(), legal_actions.end(),
                          [&](const Action &a1, const Action &a2) {
                            return (*values_)[state_i.Child(a1)] <
                                (*values_)[state_i.Child(a2)];
                          });
    Reward greedy_value = (*values_)[state_i.Child(greedy_action)];
    int num_greedy_actions =
        std::count_if(legal_actions.begin(), legal_actions.end(),
                      [&](const Action &action) {
                        return (*values_)[state_i.Child(action)] ==
                            greedy_value;
                      });
    double expectation = 0.0;
    std::vector<State> next_states = state_i.Children();
    for (const auto &state : next_states) {
      if (state_i.Child(action_i) != state) {
        if ((*values_)[state] == greedy_value)
          expectation += (*values_)[state] / num_greedy_actions;
      } else {
        if ((*values_)[state] == greedy_value)
          expectation += ret / num_greedy_actions;
      }
    }
    ret = reward_i + gamma_ * expectation;
  }
  (*values_)[update_state] += alpha_ * (ret - (*values_)[update_state]);
}

}  // namespace nim_rl
