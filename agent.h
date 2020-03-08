//
// Created by 周梓康 on 2020/3/2.
//

#ifndef NIM_AGENT_H_
#define NIM_AGENT_H_

#include <random>
#include <set>

#include "state.h"

class Game;

class Agent {
  friend void swap(Game &, Game &);
  friend class Game;

 public:
  Agent() = default;

  Agent(const Agent &) = delete;

  Agent(Agent &&) noexcept;

  Agent &operator=(const Agent &) = delete;

  Agent &operator=(Agent &&) noexcept;

  virtual ~Agent();

  virtual void Policy(Game *) = 0;

  virtual Agent *clone() const & = 0;

  virtual Agent *clone() && = 0;

 protected:
  std::set<Game *> games_;

 private:
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

  void Policy(Game *) override {}

 private:
  //static std::default_random_engine generator;
  //static std::uniform_int_distribution<unsigned> distribution(0, 6);
};

class HumanAgent : public Agent {
 public:
  void Policy(Game *) override {}
};

class OptimalAgent : public Agent {
 public:
  void Policy(Game *) override {}
};

class QAgent : public Agent {
 public:
  void Policy(Game *) override {}
};

#endif  // NIM_AGENT_H_
