//
// Created by 周梓康 on 2020/3/3.
//

#include "game.h"
#include "agent.h"

Game::Game(const State &state) : state_(state) {}

Game::Game(const State &state, Agent *first_player, Agent *second_player)
    : state_(state),
      first_player_(first_player),
      second_player_(second_player) {
  AddToAgents();
}

Game::Game(const Game &game)
    : state_(game.state_),
      first_player_(game.first_player_),
      second_player_(game.second_player_) {
  AddToAgents();
}

Game::Game(Game &&game) noexcept : state_(std::move(game.state_)) {
  MoveAgents(&game);
}

Game &Game::operator=(const Game &rhs) {
  RemoveFromAgents();
  state_ = rhs.state_;
  first_player_ = rhs.first_player_;
  second_player_ = rhs.second_player_;
  AddToAgents();
  return *this;
}

Game &Game::operator=(Game &&rhs) noexcept {
  if (this != &rhs) {
    RemoveFromAgents();
    state_ = std::move(rhs.state_);
    MoveAgents(&rhs);
  }
  return *this;
}

Game::~Game() {
  RemoveFromAgents();
}

std::vector<State::size_type> Game::NonEmptyPiles() const {
  std::vector<State::size_type> piles;
  for (decltype(state_.size()) pile_id = 0; pile_id != state_.size();
       ++pile_id) {
    if (state_[pile_id] >= 1) {
      piles.push_back(pile_id);
    }
  }
  return piles;
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
    throw std::runtime_error("Agent should not be nullptr");
  }
  do {
    first_player_->Policy(this);
    if (GameOver()) {
      break;
    }
    second_player_->Policy(this);
  } while (!GameOver());
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

void Game::AddToAgents() {
  if (first_player_) {
    first_player_->AddGame(this);
  }
  if (second_player_) {
    second_player_->AddGame(this);
  }
}

void Game::MoveAgents(Game *moved_from) {
  first_player_ = moved_from->first_player_;
  second_player_ = moved_from->second_player_;
  moved_from->RemoveFromAgents();
  AddToAgents();
  moved_from->state_.clear();
  moved_from->first_player_ = moved_from->second_player_ = nullptr;
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
  swap(lhs.first_player_, rhs.first_player_);
  swap(lhs.second_player_, rhs.second_player_);
  lhs.AddToAgents();
  rhs.AddToAgents();
}
