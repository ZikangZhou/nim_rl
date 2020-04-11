//
// Created by 周梓康 on 2020/3/3.
//

#include "nim_rl/game.h"

namespace nim_rl {

Game::Game(State state) : state_(std::move(state)) {
  initial_state_ = state_;
  all_states_ = initial_state_.GetAllStates();
}

Game::Game(State state, Agent *first_player, Agent *second_player)
    : state_(std::move(state)),
      first_player_(first_player),
      second_player_(second_player) {
  initial_state_ = state_;
  all_states_ = initial_state_.GetAllStates();
  AddToAgents();
}

Game::Game(const Game &game)
    : state_(game.state_),
      initial_state_(game.initial_state_),
      all_states_(game.all_states_),
      reward_(game.reward_),
      first_player_(game.first_player_),
      second_player_(game.second_player_) { AddToAgents(); }

Game::Game(Game &&game) noexcept
    : state_(std::move(game.state_)),
      initial_state_(std::move(game.initial_state_)),
      all_states_(std::move(game.all_states_)),
      reward_(game.reward_) { MoveAgents(&game); }

Game &Game::operator=(const Game &rhs) {
  if (this != &rhs) {
    RemoveFromAgents();
    state_ = rhs.state_;
    initial_state_ = rhs.initial_state_;
    all_states_ = rhs.all_states_;
    reward_ = rhs.reward_;
    first_player_ = rhs.first_player_;
    second_player_ = rhs.second_player_;
    AddToAgents();
  }
  return *this;
}

Game &Game::operator=(Game &&rhs) noexcept {
  if (this != &rhs) {
    RemoveFromAgents();
    state_ = std::move(rhs.state_);
    initial_state_ = std::move(rhs.initial_state_);
    all_states_ = std::move(rhs.all_states_);
    reward_ = rhs.reward_;
    MoveAgents(&rhs);
  }
  return *this;
}

void Game::Play(int episodes) {
  if (episodes < 0) {
    throw std::invalid_argument("Episodes must >= 0");
  }
  if (!first_player_ || !second_player_) {
    throw std::runtime_error("Agent should not be nullptr");
  }
  if (state_.IsEmpty()) {
    throw std::runtime_error("State should not be empty");
  }
  double win_first_player = 0.0, win_second_player = 0.0;
  Action action;
  bool play_with_human = typeid(*first_player_) == typeid(HumanAgent)
      || typeid(*second_player_) == typeid(HumanAgent);
  Reset();
  for (int i = 0; i < episodes; ++i) {
    if (play_with_human) std::cout << "Game started." << std::endl;
    while (true) {
      if (play_with_human) Render();
      action = first_player_->Step(this, true);
      if (play_with_human) {
        std::cout << "Player 1 takes action: " << action << std::endl;
        Render();
      }
      if (IsTerminal()) {
        if (reward_ == kWinReward) {
          win_first_player += 1.0;
          if (play_with_human) {
            std::cout << "Game Over. Player 1 wins." << std::endl;
          }
        }
        if (reward_ == kLoseReward) {
          win_second_player += 1.0;
          if (play_with_human) {
            std::cout << "Game Over. Player 2 wins." << std::endl;
          }
        }
        break;
      }
      action = second_player_->Step(this, true);
      if (play_with_human) {
        std::cout << "Player 2 takes action: " << action << std::endl;
      }
      if (IsTerminal()) {
        win_second_player += 1.0;
        if (play_with_human) {
          Render();
          std::cout << "Game Over. Player 2 wins." << std::endl;
        }
        break;
      }
    }
    Reset();
  }
  std::cout << std::fixed << std::setprecision(kPrecision)
            << "player 1 winning percentage: " << win_first_player / episodes
            << ", player 2 winning percentage: " << win_second_player / episodes
            << std::endl;
}

void Game::Reset() {
  state_ = initial_state_;
  reward_ = 0.0;
  first_player_->Reset();
  second_player_->Reset();
  second_player_->current_state_ = state_;
}

void Game::SetFirstPlayer(Agent *first_player) {
  if (first_player_) {
    first_player_->RemoveGame(this);
  }
  first_player_ = first_player;
  if (first_player_) {
    first_player_->AddGame(this);
  }
}

void Game::SetSecondPlayer(Agent *second_player) {
  if (second_player_) {
    second_player_->RemoveGame(this);
  }
  second_player_ = second_player;
  if (second_player_) {
    second_player_->AddGame(this);
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

void Game::Train(int episodes) {
  if (episodes < 0) {
    throw std::invalid_argument("Episodes must >= 0");
  }
  if (!first_player_ || !second_player_) {
    throw std::runtime_error("Agent should not be nullptr");
  }
  if (state_.IsEmpty()) {
    throw std::runtime_error("State should not be empty");
  }
  Reset();
  first_player_->Initialize(all_states_);
  second_player_->Initialize(all_states_);
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
    if ((i + 1) % kCheckPoint == 0) {
      if (auto first_player =
          dynamic_cast<EpsilonGreedyPolicy *>(first_player_)) {
        first_player->UpdateEpsilon();
      }
      if (auto second_player =
          dynamic_cast<EpsilonGreedyPolicy *>(second_player_)) {
        second_player->UpdateEpsilon();
      }
      std::cout << std::fixed << std::setprecision(kPrecision) << "Epoch "
                << i + 1 << ":";
      if (auto first_player = dynamic_cast<RLAgent *>(first_player_)) {
        std::cout << " player 1 optimal actions ratio: "
                  << first_player->OptimalActionsRatio();
      }
      if (auto second_player = dynamic_cast<RLAgent *>(second_player_)) {
        std::cout << " player 2 optimal actions ratio: "
                  << second_player->OptimalActionsRatio();
      }
      std::cout << std::endl;
    }
    Reset();
  }
}

void Game::AddToAgents() {
  if (first_player_) {
    first_player_->AddGame(this);
  }
  if (second_player_) {
    second_player_->AddGame(this);
  }
}

void Game::MoveAgents(Game *moved_from) {
  if (moved_from) {
    first_player_ = moved_from->first_player_;
    second_player_ = moved_from->second_player_;
    moved_from->RemoveFromAgents();
    AddToAgents();
    moved_from->state_.Clear();
    moved_from->initial_state_.Clear();
    moved_from->all_states_.clear();
    moved_from->first_player_ = moved_from->second_player_ = nullptr;
  }
}

void Game::RemoveFromAgents() {
  if (first_player_) {
    first_player_->RemoveGame(this);
  }
  if (second_player_) {
    second_player_->RemoveGame(this);
  }
}

void swap(Game &lhs, Game &rhs) {
  using std::swap;
  lhs.RemoveFromAgents();
  rhs.RemoveFromAgents();
  swap(lhs.state_, rhs.state_);
  swap(lhs.initial_state_, rhs.initial_state_);
  swap(lhs.all_states_, rhs.all_states_);
  swap(lhs.reward_, rhs.reward_);
  swap(lhs.first_player_, rhs.first_player_);
  swap(lhs.second_player_, rhs.second_player_);
  lhs.AddToAgents();
  rhs.AddToAgents();
}

}  // namespace nim_rl
