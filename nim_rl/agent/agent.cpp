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

#include "nim_rl/agent/agent.h"
#include "nim_rl/environment/game.h"

namespace nim_rl {

Agent &Agent::operator=(Agent &&rhs) noexcept {
  current_state_ = std::move(rhs.current_state_);
  return *this;
}

Action Agent::Step(Game *game, bool is_evaluation) {
  Action action = Policy(game->GetState(), is_evaluation);
  game->Step(action);
  return action;
}

Action SampleAction(const std::vector<Action> &actions) {
  static std::mt19937 rng{std::random_device{}()};
  if (actions.empty()) {
    return Action{};
  } else {
    std::uniform_int_distribution<decltype(actions.size())>
        dist(0, actions.size() - 1);
    return actions[dist(rng)];
  }
}

State SampleState(const std::vector<State> &states) {
  static std::mt19937 rng{std::random_device{}()};
  if (states.empty()) {
    return State{};
  } else {
    std::uniform_int_distribution<decltype(states.size())>
        dist(0, states.size() - 1);
    return states[dist(rng)];
  }
}

}  // namespace nim_rl
