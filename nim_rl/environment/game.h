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

#ifndef NIM_RL_ENVIRONMENT_GAME_H_
#define NIM_RL_ENVIRONMENT_GAME_H_

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "nim_rl/action/action.h"
#include "nim_rl/agent/agent.h"
#include "nim_rl/state/state.h"

namespace nim_rl {

constexpr int kCheckPoint = 1000;
constexpr double kWinReward = 1.0;
constexpr double kTieReward = 0.0;
constexpr double kLoseReward = -1.0;
constexpr double kMaxValue = 1.0;
constexpr double kMinValue = -1.0;
constexpr int kPrecision = 4;

class Game {
  friend void swap(Game &, Game &);
  friend class Agent;

 public:
  using Reward = double;
  Game() = default;
  template<typename T,
      typename = std::enable_if_t<std::is_convertible<T, State>::value>>
  explicit Game(T &&state);
  template<typename T1, typename T2, typename T3>
  Game(T1 &&state, T2 &&first_player, T3 &&second_player);
  Game(const Game &) = default;
  Game(Game &&game) noexcept
      : initial_state_(std::move(game.initial_state_)),
        state_(std::move(game.state_)),
        all_states_(std::move(game.all_states_)),
        reward_(game.reward_),
        first_player_(std::move(game.first_player_)),
        second_player_(std::move(game.second_player_)) {}
  Game &operator=(const Game &);
  Game &operator=(Game &&) noexcept;
  ~Game() = default;
  std::vector<State> GetAllStates() const { return all_states_; }
  std::shared_ptr<Agent> GetFirstPlayer() const { return first_player_; }
  State GetInitialState() const { return initial_state_; }
  Reward GetReward() const { return reward_; }
  std::shared_ptr<Agent> GetSecondPlayer() const { return second_player_; }
  State GetState() const { return state_; }
  bool IsTerminal() const { return state_.IsTerminal(); }
  void Play(int episodes = 1);
  void PrintValues() const;
  void Render() const { std::cout << "Current state: " << state_ << std::endl; }
  void Reset();
  template<typename T>
  void SetFirstPlayer(T &&first_player) {
    first_player_ = std::forward<T>(first_player).Clone();
  }
  template<typename T>
  void SetInitialState(T &&init_state) {
    initial_state_ = std::forward<T>(init_state);
  }
  void SetReward(Reward reward) { reward_ = reward; }
  template<typename T>
  void SetSecondPlayer(T &&second_player) {
    second_player_ = std::forward<T>(second_player).Clone();
  }
  template<typename T>
  void SetState(T &&state) { state_ = std::forward<T>(state); }
  void Step(const Action &);
  void Train(int episodes = 0);

 private:
  State initial_state_;
  State state_;
  std::vector<State> all_states_;
  Reward reward_ = 0.0;
  std::shared_ptr<Agent> first_player_;
  std::shared_ptr<Agent> second_player_;
};

template<typename T, typename>
Game::Game(T &&state) : state_(std::forward<T>(state)) {
  initial_state_ = state_;
  all_states_ = initial_state_.GetAllStates();
}

template<typename T1, typename T2, typename T3>
Game::Game(T1 &&state, T2 &&first_player, T3 &&second_player)
    : state_(std::forward<T1>(state)),
      first_player_(std::forward<T2>(first_player).Clone()),
      second_player_(std::forward<T3>(second_player).Clone()) {
  initial_state_ = state_;
  all_states_ = initial_state_.GetAllStates();
}

void swap(Game &, Game &);

}  // namespace nim_rl

#endif  // NIM_RL_ENVIRONMENT_GAME_H_
