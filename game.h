//
// Created by 周梓康 on 2020/3/2.
//

#ifndef NIM_GAME_H_
#define NIM_GAME_H_

#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "player.h"
#include "state.h"

class Game {
  friend void swap(Game &, Game &);

 public:
  Game() = default;

  explicit Game(const State &state);

  Game(State state, std::shared_ptr<Player> first_player,
       std::shared_ptr<Player> second_player);

  Game(const Game &) = default;

  Game(Game &&game) noexcept;

  Game &operator=(Game game);

  ~Game() = default;

  bool GameOver() const;

  std::vector<unsigned> &get_state() { return state_.get(); }

  const std::vector<unsigned> &get_state() const { return state_.get(); }

  unsigned &get_state(State::size_type pile_id) {
    return state_[pile_id];
  }

  const unsigned &get_state(State::size_type pile_id) const {
    return state_[pile_id];
  }

  void set_first_player(std::shared_ptr<Player> first_player) {
    first_player_ = std::move(first_player);
  }

  void set_second_player(std::shared_ptr<Player> second_player) {
    second_player_ = std::move(second_player);
  }

  void set_state(State::size_type pile_id, unsigned num_objects) {
    state_[pile_id] = num_objects;
  }

  void set_state(const std::vector<unsigned> &state) { state_ = state; }

  void Run();

 private:
  State state_;
  std::shared_ptr<Player> first_player_;
  std::shared_ptr<Player> second_player_;
};

inline void swap(Game &lhs, Game &rhs) {
  using std::swap;
  swap(lhs.state_, rhs.state_);
  swap(lhs.first_player_, rhs.first_player_);
  swap(lhs.second_player_, rhs.second_player_);
}

#endif  // NIM_GAME_H_
