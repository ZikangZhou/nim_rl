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

constexpr int kCheckPoint = 1000;
constexpr double kWinReward = 1.0;
constexpr double kTieReward = 0.0;
constexpr double kLoseReward = -1.0;
constexpr double kMaxValue = 1.0;
constexpr double kMinValue = -1.0;
constexpr int kPrecision = 4;

class Game {
  friend void swap(Game &, Game &);
  friend class Agent;

 public:
  using Reward = double;
  Game() = default;
  explicit Game(State state);
  Game(State state, Agent *first_player, Agent *second_player);
  Game(const Game &);
  Game(Game &&) noexcept;
  Game &operator=(const Game &);
  Game &operator=(Game &&) noexcept;
  ~Game() { RemoveFromAgents(); }
  std::vector<State> GetAllStates() const { return all_states_; }
  Agent *GetFirstPlayer() { return first_player_; }
  const Agent *GetFirstPlayer() const { return first_player_; }
  State GetInitialState() const { return initial_state_; }
  Reward GetReward() const { return reward_; }
  Agent *GetSecondPlayer() { return second_player_; }
  const Agent *GetSecondPlayer() const { return second_player_; }
  State GetState() const { return state_; }
  bool IsTerminal() const { return state_.IsTerminal(); }
  void Play(int episodes = 1);
  void Render() const { std::cout << "Current state: " << state_ << std::endl; }
  void Reset();
  void SetFirstPlayer(Agent *);
  template<typename T>
  void SetInitialState(T &&init_state) {
    initial_state_ = std::forward<T>(init_state);
  }
  void SetReward(Reward reward) { reward_ = reward; }
  void SetSecondPlayer(Agent *);
  template<typename T>
  void SetState(T &&state) { state_ = std::forward<T>(state); }
  void Step(const Action &);
  void Train(int episodes = 0);

 private:
  State initial_state_;
  State state_;
  std::vector<State> all_states_;
  Reward reward_ = 0.0;
  Agent *first_player_ = nullptr;
  Agent *second_player_ = nullptr;
  void AddToAgents();
  void MoveAgents(Game *);
  void RemoveFromAgents();
};

void swap(Game &, Game &);

#endif  // NIM_GAME_H_
