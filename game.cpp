//
// Created by 周梓康 on 2020/3/3.
//

#include "game.h"

Game::Game(const State &state) : state_(state), first_player_(nullptr),
                                 second_player_(nullptr) {
  if (state_.empty()) {
    throw std::invalid_argument("State should not be empty");
  }
}

Game::Game(const State &state, Player *first_player, Player *second_player)
    : state_(state), first_player_(first_player),
      second_player_(second_player) {
  if (state_.empty()) {
    throw std::invalid_argument("State should not be empty");
  }
}

Game::~Game() {
  delete first_player_;
  delete second_player_;
}

bool Game::GameOver() const {
  for (decltype(state_.size()) pile_id = 0; pile_id != state_.size();
       ++pile_id) {
    if (state_[pile_id]) {
      return false;
    }
  }
  return true;
}

void Game::Run() {
  do {
    first_player_->Action(&state_);
    if (GameOver()) {
      break;
    }
    second_player_->Action(&state_);
  } while (!GameOver());
}
