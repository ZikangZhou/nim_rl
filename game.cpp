//
// Created by 周梓康 on 2020/3/3.
//

#include "game.h"

Game::Game(const State &state) : state_(state) {
  if (state_.empty()) {
    std::cerr << "Warning: State is empty." << std::endl;
  }
}

Game::Game(State state, std::shared_ptr<Player> first_player,
           std::shared_ptr<Player> second_player) :
    state_(std::move(state)),
    first_player_(std::move(first_player)),
    second_player_(std::move(second_player)) {
  if (state_.empty()) {
    std::cerr << "Warning: State is empty." << std::endl;
  }
}

Game::Game(Game &&game) noexcept :
    state_(std::move(game.state_)),
    first_player_(std::move(game.first_player_)),
    second_player_(std::move(game.second_player_)) {
  game.state_ = {};
  game.first_player_.reset();
  game.second_player_.reset();
}

Game &Game::operator=(Game game) {
  swap(*this, game);
  return *this;
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
  if (!first_player_ || !second_player_) {
    throw std::runtime_error("Player should not be nullptr");
  }
  do {
    first_player_->Action(&state_);
    if (GameOver()) {
      break;
    }
    second_player_->Action(&state_);
  } while (!GameOver());
}
