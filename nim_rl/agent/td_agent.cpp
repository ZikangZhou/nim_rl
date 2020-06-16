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

#include "nim_rl/agent/td_agent.h"
#include "nim_rl/environment/game.h"

namespace nim_rl {

Action TDAgent::Step(Game *game, bool is_evaluation) {
  Reward reward = game->GetReward();
  Action action = Agent::Step(game, is_evaluation);
  if (!is_evaluation) Update(current_state_, game->GetState(), -reward);
  return action;
}

void QLearningAgent::Update(const State &update_state,
                            const State &current_state,
                            Reward reward) {
  if (!update_state.IsEmpty())
    (*values_)[update_state] +=
        alpha_ * (reward + gamma_ * greedy_value_ - (*values_)[update_state]);
  current_state_ = current_state;
}

void SarsaAgent::Update(const State &update_state,
                        const State &current_state,
                        Reward reward) {
  if (!update_state.IsEmpty())
    (*values_)[update_state] += alpha_
        * (reward + gamma_ * (*values_)[current_state]
            - (*values_)[update_state]);
  current_state_ = current_state;
}

Action ExpectedSarsaAgent::Policy(const State &state, bool is_evaluation) {
  SetNextStates(state.Children());
  return TDAgent::Policy(state, is_evaluation);
}

void ExpectedSarsaAgent::Reset() {
  TDAgent::Reset();
  next_states_.clear();
}

void ExpectedSarsaAgent::Update(const State &update_state,
                                const State &current_state,
                                Reward reward) {
  double epsilon = epsilon_greedy_.GetEpsilon();
  if (!update_state.IsEmpty()) {
    double expectation = 0.0;
    if (!legal_actions_.empty()) {
      for (const auto &state : next_states_)
        if ((*values_)[state] != greedy_value_)
          expectation += epsilon * (*values_)[state] / legal_actions_.size();
      expectation += (1 - epsilon) * greedy_value_
          + greedy_actions_.size() * epsilon * greedy_value_
              / legal_actions_.size();
    }
    (*values_)[update_state] +=
        alpha_ * (reward + gamma_ * expectation - (*values_)[update_state]);
  }
  current_state_ = current_state;
}

std::unordered_map<State, Agent::Reward>
DoubleLearningAgent::GetValues() const {
  Values values = Values(*values_);
  for (const auto &kv : *values_2_)
    values[kv.first] = (values[kv.first] + kv.second) / 2;
  return values;
}

void
DoubleLearningAgent::Initialize(const std::vector<State> &all_states) {
  TDAgent::Initialize(all_states);
  *values_2_ = *values_;
}

Action DoubleLearningAgent::Policy(const State &state, bool is_evaluation) {
  legal_actions_ = state.LegalActions();
  greedy_actions_.clear();
  if (legal_actions_.empty()) {
    greedy_value_ = 0.0;
    return Action{};
  } else {
    Action greedy_action =
        *std::max_element(legal_actions_.begin(), legal_actions_.end(),
                          [&](const Action &a1, const Action &a2) -> bool {
                            State state1 = state.Child(a1);
                            State state2 = state.Child(a2);
                            return (*values_)[state1] + (*values_2_)[state1] <
                                (*values_)[state2] + (*values_2_)[state2];
                          });
    State greedy_state = state.Child(greedy_action);
    greedy_value_ = ((*values_)[greedy_state] + (*values_2_)[greedy_state]) / 2;
    std::copy_if(legal_actions_.begin(), legal_actions_.end(),
                 std::back_inserter(greedy_actions_),
                 [&state, this](const Action &action) -> bool {
                   State next_state = state.Child(action);
                   return
                       ((*values_)[next_state] + (*values_2_)[next_state]) / 2
                           == greedy_value_;
                 });
    flag_ = dist_flag_(rng_);
    if (is_evaluation) {
      return SampleAction(greedy_actions_);
    } else {
      return PolicyImpl(legal_actions_, greedy_actions_);
    }
  }
}

void DoubleLearningAgent::Reset() {
  TDAgent::Reset();
  flag_ = false;
}

void DoubleLearningAgent::Update(const State &update_state,
                                 const State &current_state,
                                 Reward reward) {
  flag_ ? DoUpdate(update_state, current_state, reward, values_.get())
        : DoUpdate(update_state, current_state, reward, values_2_.get());
  current_state_ = current_state;
}

void DoubleQLearningAgent::DoUpdate(const State &update_state,
                                    const State &/*current_state*/,
                                    Reward reward,
                                    Values *values) {
  if (!update_state.IsEmpty())
    (*values)[update_state] += alpha_
        * (reward + gamma_ * greedy_value_ - (*values)[update_state]);
}

Action DoubleQLearningAgent::Policy(const State &state,
                                        bool is_evaluation) {
  Action action = DoubleLearningAgent::Policy(state, is_evaluation);
  greedy_actions_.clear();
  if (!legal_actions_.empty()) {
    Action greedy_action;
    if (flag_) {
      greedy_action =
          *std::max_element(legal_actions_.begin(), legal_actions_.end(),
                            [&](const Action &a1, const Action &a2) {
                              return (*values_)[state.Child(a1)] <
                                  (*values_)[state.Child(a2)];
                            });
      greedy_value_ = (*values_2_)[state.Child(greedy_action)];
    } else {
      greedy_action =
          *std::max_element(legal_actions_.begin(), legal_actions_.end(),
                            [&](const Action &a1, const Action &a2) {
                              return (*values_2_)[state.Child(a1)] <
                                  (*values_2_)[state.Child(a2)];
                            });
      greedy_value_ = (*values_)[state.Child(greedy_action)];
    }
  }
  return action;
}

void DoubleSarsaAgent::DoUpdate(const State &update_state,
                                const State &current_state,
                                Reward reward,
                                Values *values) {
  if (!update_state.IsEmpty()) {
    if (values == values_.get()) {
      (*values)[update_state] += alpha_
          * (reward + gamma_ * (*values_2_)[current_state]
              - (*values)[update_state]);
    } else {
      (*values)[update_state] += alpha_
          * (reward + gamma_ * (*values_)[current_state]
              - (*values)[update_state]);
    }
  }
}

void DoubleExpectedSarsaAgent::DoUpdate(const State &update_state,
                                        const State &/*current_state*/,
                                        Reward reward,
                                        Values *values) {
  double epsilon = epsilon_greedy_.GetEpsilon();
  if (!update_state.IsEmpty()) {
    double expectation = 0.0;
    if (!legal_actions_.empty()) {
      if (values == values_.get()) {
        for (const auto &state : next_states_)
          if ((*values_2_)[state] != greedy_value_)
            expectation +=
                (*values_2_)[state] * epsilon / legal_actions_.size();
      } else if (values == values_2_.get()) {
        for (const auto &state : next_states_)
          if ((*values_)[state] != greedy_value_)
            expectation += (*values_)[state] * epsilon / legal_actions_.size();
      }
      expectation += (1 - epsilon) * greedy_value_
          + greedy_actions_.size() * epsilon * greedy_value_
              / legal_actions_.size();
    }
    (*values)[update_state] +=
        alpha_ * (reward + gamma_ * expectation - (*values)[update_state]);
  }
}

Action DoubleExpectedSarsaAgent::Policy(const State &state,
                                        bool is_evaluation) {
  SetNextStates(state.Children());
  Action action = DoubleLearningAgent::Policy(state, is_evaluation);
  greedy_actions_.clear();
  if (!legal_actions_.empty()) {
    Action greedy_action =
        *std::max_element(legal_actions_.begin(), legal_actions_.end(),
                          [&](const Action &a1, const Action &a2) {
                            State state1 = state.Child(a1);
                            State state2 = state.Child(a2);
                            return (*values_)[state1] + (*values_2_)[state1] <
                                (*values_)[state2] + (*values_2_)[state2];
                          });
    greedy_value_ = ((*values_)[state.Child(greedy_action)]
        + (*values_2_)[state.Child(greedy_action)]) / 2;
    std::copy_if(legal_actions_.begin(), legal_actions_.end(),
                 std::back_inserter(greedy_actions_),
                 [&](const Action &action) {
                   State next_state = state.Child(action);
                   return
                       ((*values_)[next_state] + (*values_2_)[next_state]) / 2
                           == greedy_value_;
                 });
  }
  return action;
}

void DoubleExpectedSarsaAgent::Reset() {
  DoubleLearningAgent::Reset();
  next_states_.clear();
}

}  // namespace nim_rl
