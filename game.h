//
// Created by 周梓康 on 2020/3/2.
//

#ifndef NIM_GAME_H_
#define NIM_GAME_H_

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "action.h"
#include "agent.h"
#include "state.h"

class Game {
  friend void swap(Game &, Game &);
  friend class Agent;

 public:
  using Reward = double;
  Game() = default;
  explicit Game(State state) : state_(std::move(state)) {
    init_state_ = state_;
  }
  Game(State state, Agent *first_player, Agent *second_player);
  Game(const Game &);
  Game(Game &&) noexcept;
  Game &operator=(const Game &);
  Game &operator=(Game &&) noexcept;
  ~Game();
  Agent *first_player() { return first_player_; }
  const Agent *first_player() const { return first_player_; }
  bool GameOver() const;
  State &init_state() { return init_state_; }
  const State &init_state() const { return init_state_; }
  void Render() { std::cout << "Current state: " << state_ << std::endl; }
  void Reset();
  Reward reward() { return reward_; }
  const Reward reward() const { return reward_; }
  void Play(int episodes = 1);
  Agent *second_player() { return second_player_; }
  const Agent *second_player() const { return second_player_; }
  void set_first_player(Agent *);
  template<typename T>
  void set_init_state(T &&init_state) {
    init_state_ = std::forward<T>(init_state);
  }
  void set_reward(Reward reward) { reward_ = reward; }
  void set_second_player(Agent *);
  template<typename T>
  void set_state(T &&state) { state_ = std::forward<T>(state); }
  State &state() { return state_; }
  const State &state() const { return state_; }
  void Step(const Action &action, State *state, Reward *reward, bool *done);
  void Train(int episodes);

 private:
  State state_;
  State init_state_;
  Reward reward_ = 0.0;
  Agent *first_player_ = nullptr;
  Agent *second_player_ = nullptr;
  void AddToAgents();
  void MoveAgents(Game *);
  void RemoveFromAgents();
};

void swap(Game &, Game &);

#endif  // NIM_GAME_H_
