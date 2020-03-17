//
// Created by 周梓康 on 2020/3/3.
//

#include "game.h"

Game::Game(State state, Agent *first_player, Agent *second_player)
    : state_(std::move(state)),
      first_player_(first_player),
      second_player_(second_player) {
  init_state_ = state_;
  AddToAgents();
}

Game::Game(const Game &game)
    : state_(game.state_),
      init_state_(game.init_state_),
      reward_(game.reward_),
      first_player_(game.first_player_),
      second_player_(game.second_player_) {
  AddToAgents();
}

Game::Game(Game &&game) noexcept
    : state_(std::move(game.state_)),
      init_state_(std::move(game.init_state_)),
      reward_(game.reward_) {
  MoveAgents(&game);
}

Game &Game::operator=(const Game &rhs) {
  RemoveFromAgents();
  state_ = rhs.state_;
  init_state_ = rhs.init_state_;
  reward_ = rhs.reward_;
  first_player_ = rhs.first_player_;
  second_player_ = rhs.second_player_;
  AddToAgents();
  return *this;
}

Game &Game::operator=(Game &&rhs) noexcept {
  if (this != &rhs) {
    RemoveFromAgents();
    state_ = std::move(rhs.state_);
    init_state_ = std::move(rhs.init_state_);
    reward_ = rhs.reward_;
    MoveAgents(&rhs);
  }
  return *this;
}

Game::~Game() {
  RemoveFromAgents();
}

bool Game::GameOver() const {
  for (int pile_id = 0; pile_id != state_.Size();
       ++pile_id) {
    if (state_[pile_id]) {
      return false;
    }
  }
  return true;
}

void Game::Reset() {
  state_ = init_state_;
  reward_ = 0.0;
}

Agent *Game::Play() {
  if (!first_player_ || !second_player_) {
    throw std::runtime_error("Agent should not be nullptr");
  }
  if (state_.Empty()) {
    throw std::runtime_error("State should not be empty");
  }
  Action action_first_player, action_second_player;
  std::cout << "Game started." << std::endl;
  Reset();
  do {
    Render();
    action_first_player = first_player_->Policy(state_);
    state_.Update(action_first_player);
    std::cout << "Player 1 takes action: " << action_first_player << std::endl;
    Render();
    if (GameOver()) {
      std::cout << "Game Over. Player 1 wins." << std::endl;
      return first_player_;
    }
    action_second_player = second_player_->Policy(state_);
    state_.Update(action_second_player);
    std::cout << "Player 2 takes action: " << action_second_player << std::endl;
  } while (!GameOver());
  std::cout << "Current state: " << state_ << std::endl;
  std::cout << "Game over. Player 2 wins." << std::endl;
  return second_player_;
}

void Game::set_first_player(Agent *first_player) {
  if (first_player_) {
    first_player_->RemoveGame(this);
  }
  first_player_ = first_player;
  if (first_player_) {
    first_player_->AddGame(this);
  }
}

void Game::set_second_player(Agent *second_player) {
  if (second_player_) {
    second_player_->RemoveGame(this);
  }
  second_player_ = second_player;
  if (second_player_) {
    second_player_->AddGame(this);
  }
}

void Game::Step(const Action &action,
                State *state,
                Reward *reward,
                bool *done) {
  if (action.Valid(state_)) {
    state_.Update(action);
    if (GameOver()) {
      reward_ = 1.0;
      *done = true;
    } else {
      reward_ = 0.0;
      *done = false;
    }
  } else {
    reward_ = -1.0;
    *done = true;
  }
  *state = state_;
  *reward = reward_;
}

void Game::Train(int episodes) {
  if (episodes <= 0) {
    throw std::invalid_argument("Episodes must be positive");
  }
  if (!first_player_ || !second_player_) {
    throw std::runtime_error("Agent should not be nullptr");
  }
  if (state_.Empty()) {
    throw std::runtime_error("State should not be empty");
  }
  double win_first_player = 0.0, win_second_player = 0.0;
  Action action_first_player, action_second_player;
  State next_state_first_player, next_state_second_player;
  Reward reward_first_player = 0.0, reward_second_player = 0.0;
  bool done = false;
  Reset();
  first_player_->current_state_ = State();
  second_player_->current_state_ = state_;
  for (int i = 0; i < episodes; ++i) {
    while (true) {
      action_first_player = first_player_->Policy(state_);
      Step(action_first_player,
           &next_state_first_player,
           &reward_first_player,
           &done);
      first_player_->Update(first_player_->current_state_,
                            next_state_first_player,
                            reward_first_player);
      first_player_->current_state_ = next_state_first_player;
      if (done) {
        second_player_->Update(second_player_->current_state_,
                               first_player_->current_state_,
                               -reward_first_player);
        break;
      }
      action_second_player =
          second_player_->Policy(state_);
      Step(action_second_player,
           &next_state_second_player,
           &reward_second_player,
           &done);
      second_player_->Update(second_player_->current_state_,
                             next_state_second_player,
                             reward_second_player);
      second_player_->current_state_ = next_state_second_player;
      if (done) {
        first_player_->Update(first_player_->current_state_,
                              second_player_->current_state_,
                              -reward_second_player);
        break;
      }
    }
    if (reward_first_player == 1.0) {
      win_first_player += 1;
    }
    if (reward_second_player == 1.0) {
      win_second_player += 1;
    }
    if ((i + 1) % 1000 == 0) {
      std::cout << std::fixed << std::setprecision(4) << "Epoch " << i + 1
                << ", player 1 winning percentage: "
                << win_first_player / (i + 1)
                << ", player 2 winning percentage: "
                << win_second_player / (i + 1) << std::endl;
    }
    Reset();
    first_player_->current_state_ = State();
    second_player_->current_state_ = state_;
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
    moved_from->init_state_.Clear();
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
  swap(lhs.init_state_, rhs.init_state_);
  swap(lhs.reward_, rhs.reward_);
  swap(lhs.first_player_, rhs.first_player_);
  swap(lhs.second_player_, rhs.second_player_);
  lhs.AddToAgents();
  rhs.AddToAgents();
}
