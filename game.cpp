//
// Created by 周梓康 on 2020/3/3.
//

#include "game.h"

#include <iostream>

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

bool Game::GameOver() const {
  for (decltype(state_.size()) pile_id = 0; pile_id != state_.size();
       ++pile_id) {
    if (state_[pile_id]) {
      return false;
    }
  }
  return true;
}

std::vector<Action> Game::GetActions() const {
  std::vector<Action> actions;
  for (decltype(state_.size()) pile_id = 0; pile_id != state_.size();
       ++pile_id) {
    if (state_[pile_id]) {
      for (unsigned num_objects = 1; num_objects != state_[pile_id] + 1;
           ++num_objects) {
        actions.emplace_back(pile_id, num_objects);
      }
    }
  }
  return actions;
}

void Game::Run() {
  if (!first_player_ || !second_player_) {
    throw std::runtime_error("Agent should not be nullptr");
  }
  if (state_.empty()) {
    throw std::runtime_error("State should not be empty");
  }
  Action action1, action2;
  std::cout << "Game started." << std::endl;
  do {
    std::cout << "Current state: " << state_ << std::endl;
    action1 = first_player_->Policy(this);
    state_.TakeAction(action1);
    std::cout << "Player 1 takes action: " << action1 << std::endl;
    if (GameOver()) {
      std::cout << "Game Over." << std::endl;
      break;
    }
    std::cout << "Current state: " << state_ << std::endl;
    action2 = second_player_->Policy(this);
    state_.TakeAction(action2);
    std::cout << "Player 2 takes action: " << action2 << std::endl;
  } while (!GameOver());
  std::cout << "Game over." << std::endl;
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
