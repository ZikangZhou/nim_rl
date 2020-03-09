//
// Created by 周梓康 on 2020/3/2.
//

#ifndef NIM_AGENT_H_
#define NIM_AGENT_H_

#include <iostream>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "action.h"
#include "state.h"

class Game;

class Agent {
  friend void swap(Game &, Game &);
  friend class Game;

 public:
  Agent() = default;
  Agent(const Agent &) = delete;
  Agent(Agent &&agent) noexcept { MoveGames(&agent); }
  Agent &operator=(const Agent &) = delete;
  Agent &operator=(Agent &&) noexcept;
  virtual ~Agent() { RemoveFromGames(); }
  std::set<Game *> &games() { return games_; }
  const std::set<Game *> &games() const { return games_; }
  virtual Action Policy(Game *) = 0;

 protected:
  std::set<Game *> games_;
  Action RandomPickAction(const std::vector<Action> &);

 private:
  std::mt19937 generator_{std::random_device{}()};
  void AddGame(Game *game) { games_.insert(game); }
  void MoveGames(Agent *);
  void RemoveFromGames();
  void RemoveGame(Game *game) { games_.erase(game); }
};

class RandomAgent : public Agent {
 public:
  RandomAgent() = default;
  RandomAgent(const RandomAgent &) = delete;
  RandomAgent(RandomAgent &&) = default;
  RandomAgent &operator=(const RandomAgent &) = delete;
  RandomAgent &operator=(RandomAgent &&) = default;
  Action Policy(Game *) override;
};

class HumanAgent : public Agent {
 public:
  explicit HumanAgent(std::istream &is = std::cin, std::ostream &os = std::cout)
      : Agent(), is_(is), os_(os) {}

  HumanAgent(const HumanAgent &) = delete;
  HumanAgent(HumanAgent &&) = default;
  HumanAgent &operator=(const HumanAgent &) = delete;
  HumanAgent &operator=(HumanAgent &&) = delete;
  Action Policy(Game *) override;

 private:
  std::istream &is_;
  std::ostream &os_;
};

class OptimalAgent : public Agent {
 public:
  OptimalAgent() = default;
  OptimalAgent(const OptimalAgent &) = delete;
  OptimalAgent(OptimalAgent &&) = default;
  OptimalAgent &operator=(const OptimalAgent &) = delete;
  OptimalAgent &operator=(OptimalAgent &&) = default;
  Action Policy(Game *) override;
};

class QAgent : public Agent {
 public:
  Action Policy(Game *) override { return Action{-1, -1}; }
};

#endif  // NIM_AGENT_H_
