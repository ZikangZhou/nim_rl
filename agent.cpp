//
// Created by 周梓康 on 2020/3/3.
//

#include "agent.h"
#include "game.h"

Agent::Agent(Agent &&agent) noexcept {
  MoveGames(&agent);
}

Agent &Agent::operator=(Agent &&rhs) noexcept {
  if (this != &rhs) {
    RemoveFromGames();
    MoveGames(&rhs);
  }
  return *this;
}

Agent::~Agent() {
  RemoveFromGames();
}

void Agent::MoveGames(Agent *moved_from) {
  games_ = std::move(moved_from->games_);
  for (auto game : games_) {
    if (game->first_player_ == moved_from) {
      game->first_player_ = this;
    }
    if (game->second_player_ == moved_from) {
      game->second_player_ = this;
    }
  }
  moved_from->games_.clear();
}

void Agent::RemoveFromGames() {
  for (auto game : games_) {
    if (game->first_player_ == this) {
      game->first_player_ = nullptr;
    }
    if (game->second_player_ == this) {
      game->second_player_ = nullptr;
    }
  }
}
