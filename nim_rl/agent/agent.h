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

#ifndef NIM_RL_AGENT_AGENT_H_
#define NIM_RL_AGENT_AGENT_H_

#include <algorithm>
#include <climits>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nim_rl/action/action.h"
#include "nim_rl/state/state.h"

namespace std {
using nim_rl::State;
using nim_rl::Action;
template<>
struct hash<std::pair<State, Action>> {
  std::size_t operator()(const std::pair<State, Action> &state_action) const {
    std::size_t seed = 0;
    seed ^= std::hash<State>()(state_action.first) + 0x9e3779b9
        + (seed << 6u) + (seed >> 2u);
    seed ^= std::hash<Action>()(state_action.second) + 0x9e3779b9
        + (seed << 6u) + (seed >> 2u);
    return seed;
  }
};
}  // namespace std

namespace nim_rl {

Action SampleAction(const std::vector<Action> &);

State SampleState(const std::vector<State> &);

class Game;

class Agent : public std::enable_shared_from_this<Agent> {
  friend class Game;

 public:
  using Reward = double;
  using Value = double;
  Agent() = default;
  Agent(const Agent &) = default;
  Agent(Agent &&agent) noexcept
      : current_state_(std::move(agent.current_state_)) {}
  Agent &operator=(const Agent &) = default;
  Agent &operator=(Agent &&) noexcept;
  virtual ~Agent() = default;
  virtual std::shared_ptr<Agent> Clone() const = 0;
  State GetCurrentState() const { return current_state_; }
  virtual void Initialize(const std::vector<State> &) {}
  virtual Action Policy(const State &, bool is_evaluation) = 0;
  virtual void Reset() { current_state_ = State(); }
  void SetCurrentState(const State &state) { current_state_ = state; }
  virtual Action Step(Game *, bool is_evaluation);
  virtual void Update(const State &update_state, const State &current_state,
                      Reward reward) { current_state_ = current_state; }

 protected:
  State current_state_;
};

}  // namespace nim_rl

#endif  // NIM_RL_AGENT_AGENT_H_
