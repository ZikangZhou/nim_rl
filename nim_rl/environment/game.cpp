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

#include "nim_rl/environment/game.h"
#include "nim_rl/agent/human_agent.h"
#include "nim_rl/agent/rl_agent.h"

namespace nim_rl {

Game &Game::operator=(const Game &rhs) {
  if (this != &rhs) {
    initial_state_ = rhs.initial_state_;
    state_ = rhs.state_;
    all_states_ = rhs.all_states_;
    reward_ = rhs.reward_;
    first_player_ = rhs.first_player_;
    second_player_ = rhs.second_player_;
  }
  return *this;
}

Game &Game::operator=(Game &&rhs) noexcept {
  if (this != &rhs) {
    initial_state_ = std::move(rhs.initial_state_);
    state_ = std::move(rhs.state_);
    all_states_ = std::move(rhs.all_states_);
    reward_ = rhs.reward_;
    first_player_ = std::move(rhs.first_player_);
    second_player_ = std::move(rhs.second_player_);
  }
  return *this;
}

void Game::Play(int episodes) {
  if (episodes < 0) throw std::invalid_argument("Episodes must >= 0");
  if (!first_player_ || !second_player_)
    throw std::runtime_error("Agent should not be nullptr");
  if (state_.IsEmpty()) throw std::runtime_error("State should not be empty");
  double win_first_player = 0.0, win_second_player = 0.0;
  Action action;
  bool play_with_human = typeid(*first_player_) == typeid(HumanAgent) ||
      typeid(*second_player_) == typeid(HumanAgent);
  Reset();
  int cnt = 0;
  double average_episode_size_1 = 0.0, average_episode_size_2 = 0.0;
  for (int i = 0; i < episodes; ++i) {
    ++cnt;
    int episode_size_1 = 0, episode_size_2 = 0;
    if (play_with_human) std::cout << "Game started." << std::endl;
    while (true) {
      if (play_with_human) Render();
      ++episode_size_1;
      action = first_player_->Step(this, true);
      if (play_with_human) {
        std::cout << "Player 1 takes action: " << action << std::endl;
        Render();
      }
      if (IsTerminal()) {
        if (reward_ == kWinReward) {
          win_first_player += 1.0;
          if (play_with_human)
            std::cout << "Game Over. Player 1 wins." << std::endl;
        }
        if (reward_ == kLoseReward) {
          win_second_player += 1.0;
          if (play_with_human)
            std::cout << "Game Over. Player 2 wins." << std::endl;
        }
        break;
      }
      ++episode_size_2;
      action = second_player_->Step(this, true);
      if (play_with_human)
        std::cout << "Player 2 takes action: " << action << std::endl;
      if (IsTerminal()) {
        win_second_player += 1.0;
        if (play_with_human) {
          Render();
          std::cout << "Game Over. Player 2 wins." << std::endl;
        }
        break;
      }
    }
    average_episode_size_1 += (episode_size_1 - average_episode_size_1) / cnt;
    average_episode_size_2 += (episode_size_2 - average_episode_size_2) / cnt;
    Reset();
  }
  std::cout << "average episode size: " << average_episode_size_1 << " "
            << average_episode_size_2 << std::endl;
  std::cout << std::fixed << std::setprecision(kPrecision)
            << "player 1 winning percentage: " << win_first_player / episodes
            << ", player 2 winning percentage: " << win_second_player / episodes
            << std::endl;
}

void Game::PrintValues() const {
  if (auto first_player = dynamic_cast<RLAgent *>(first_player_.get())) {
    std::cout << "player 1 values: " << first_player->GetValues()
              << std::endl;
  }
  if (auto second_player = dynamic_cast<RLAgent *>(second_player_.get())) {
    std::cout << "player 2 values: " << second_player->GetValues()
              << std::endl;
  }
}

void Game::Reset() {
  state_ = initial_state_;
  reward_ = 0.0;
  if (first_player_) first_player_->Reset();
  if (second_player_) {
    second_player_->Reset();
    second_player_->current_state_ = state_;
  }
}

void Game::Step(const Action &action) {
  if (action.IsLegal(state_)) {
    state_.ApplyAction(action);
    reward_ = IsTerminal() ? kWinReward : kTieReward;
  } else {
    state_ = State();
    reward_ = kLoseReward;
  }
}

std::pair<std::vector<double>, std::vector<double>> Game::Train(int episodes) {
  if (episodes < 0) throw std::invalid_argument("Episodes must >= 0");
  if (!first_player_ || !second_player_)
    throw std::runtime_error("Agent should not be nullptr");
  if (state_.IsEmpty()) throw std::runtime_error("State should not be empty");
  std::vector<double> optimal_action_ratios, mean_square_errors;
  first_player_->Initialize(all_states_);
  second_player_->Initialize(all_states_);
  if (auto first_player = dynamic_cast<RLAgent *>(first_player_.get())) {
    double optimal_action_ratio = first_player->OptimalActionsRatio();
    optimal_action_ratios.push_back(optimal_action_ratio);
    mean_square_errors.push_back(first_player->MinSquareError());
    std::cout << "Epoch 1: " << optimal_action_ratio << std::endl;
  }
  Reset();
  for (int i = 0; i < episodes; ++i) {
    while (true) {
      first_player_->Step(this, false);
      if (IsTerminal()) {
        second_player_->Step(this, false);
        break;
      }
      second_player_->Step(this, false);
      if (IsTerminal()) {
        first_player_->Step(this, false);
        break;
      }
    }
    if (auto first_player = dynamic_cast<RLAgent *>(first_player_.get())) {
      first_player->UpdateExploration(i);
    }
    if (auto second_player = dynamic_cast<RLAgent *>(second_player_.get())) {
      second_player->UpdateExploration(i);
    }
    if ((i + 1) % kCheckPoint == 0) {
      std::cout << "Epoch " << i + 1 << ":";
      if (auto first_player = dynamic_cast<RLAgent *>(first_player_.get())) {
        double optimal_action_ratio = first_player->OptimalActionsRatio();
        double mean_square_error = first_player->MinSquareError();
        optimal_action_ratios.push_back(optimal_action_ratio);
        mean_square_errors.push_back(mean_square_error);
        std::cout << optimal_action_ratio << std::endl;
//        std::cout << " player 1 optimal actions ratio: "
//                  << optimal_action_ratio << ", ";
//        std::cout << "mean square error: " << first_player->MinSquareError()
//                  << std::endl;
      }
    }
    Reset();
  }
  return std::make_pair(optimal_action_ratios, mean_square_errors);
}

void swap(Game &lhs, Game &rhs) {
  using std::swap;
  swap(lhs.state_, rhs.state_);
  swap(lhs.initial_state_, rhs.initial_state_);
  swap(lhs.all_states_, rhs.all_states_);
  swap(lhs.reward_, rhs.reward_);
  swap(lhs.first_player_, rhs.first_player_);
  swap(lhs.second_player_, rhs.second_player_);
}

}  // namespace nim_rl
