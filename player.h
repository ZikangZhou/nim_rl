//
// Created by 周梓康 on 2020/3/2.
//

#ifndef NIM_PLAYER_H_
#define NIM_PLAYER_H_

#include "state.h"

class Player {
 public:
  Player(const Player &) = delete;
  Player &operator=(const Player &) = delete;
  virtual ~Player() = default;
  virtual void Action(State *state) = 0;

 protected:
  Player() = default;
};

class RandomPlayer : public Player {
 public:
  void Action(State *state) override {}
};

class HumanPlayer : public Player {
 public:
  void Action(State *state) override {}
};

class OptimalPlayer : public Player {
 public:
  void Action(State *state) override {}
};

class Agent : public Player {
 public:
  void Action(State *state) override {}
};

#endif  // NIM_PLAYER_H_
