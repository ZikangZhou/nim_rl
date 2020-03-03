//
// Created by 周梓康 on 2020/3/3.
//

#include "game.h"

Game::Game(State state, Player *first_player, Player *second_player)
        : state_(std::move(state)), first_player_(first_player), second_player_(second_player) {
    if (state_.empty()) {
        throw std::invalid_argument("State should not be empty");
    }
    if (!first_player_ || !second_player_) {
        throw std::invalid_argument("Player should not be nullptr");
    }
}

bool Game::GameOver() const {
    for (decltype(state_.size()) index = 0; index != state_.size(); ++index) {
        if (state_[index]) {
            return false;
        }
    }
    return true;
}

void Game::Run() {
    do {
        first_player_->Action(state_);
        if (GameOver()) {
            break;
        }
        second_player_->Action(state_);
    } while (!GameOver());
}
