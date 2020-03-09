//
// Created by 周梓康 on 2020/3/2.
//

#ifndef NIM_GAME_H_
#define NIM_GAME_H_

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
  Game() = default;
  explicit Game(State state) : state_(std::move(state)) {}
  Game(State state, Agent *first_player, Agent *second_player);
  Game(const Game &);
  Game(Game &&) noexcept;
  Game &operator=(const Game &);
  Game &operator=(Game &&) noexcept;
  ~Game();
  Agent &first_player() { return *first_player_; }
  const Agent &first_player() const { return *first_player_; }
  bool GameOver() const;
  std::vector<Action> GetActions() const;
  void Run();
  Agent &second_player() { return *second_player_; }
  const Agent &second_player() const { return *second_player_; }
  void set_first_player(Agent *);
  void set_second_player(Agent *);
  void set_state(const State &state) { state_ = state; }
  void set_state(State &&state) { state_ = std::move(state); }
  State &state() { return state_; }
  const State &state() const { return state_; }

 private:
  State state_;
  Agent *first_player_ = nullptr;
  Agent *second_player_ = nullptr;
  void AddToAgents();
  void MoveAgents(Game *);
  void RemoveFromAgents();
};

void swap(Game &, Game &);

#endif  // NIM_GAME_H_
