//
// Created by 周梓康 on 2020/3/2.
//

#ifndef NIM_GAME_H_
#define NIM_GAME_H_

#include <stdexcept>
#include <utility>
#include <vector>

#include "player.h"
#include "state.h"

class Game {
 public:
  Game() = default;

  explicit Game(const State &state);

  Game(const State &state, Player *first_player, Player *second_player);

  Game(const Game &) = delete;

  Game &operator=(const Game &) = delete;

  ~Game();

  bool GameOver() const;

  std::vector<unsigned> &get_state() { return state_.get(); }

  const std::vector<unsigned> &get_state() const { return state_.get(); }

  unsigned &get_state(std::vector<unsigned>::size_type pile_id) {
    return state_[pile_id];
  }

  const unsigned &get_state(std::vector<unsigned>::size_type pile_id) const {
    return state_[pile_id];
  }

  void set_first_player(Player *first_player) { first_player_ = first_player; }

  void set_second_player(Player *second_player) {
    second_player_ = second_player;
  }

  void set_state(std::vector<unsigned>::size_type pile_id,
                 unsigned num_objects) {
    state_[pile_id] = num_objects;
  }

  void set_state(const std::vector<unsigned> &state) { state_ = state; }

  void Run();

 private:
  State state_;
  Player *first_player_ = nullptr;
  Player *second_player_ = nullptr;
};

#endif  // NIM_GAME_H_
