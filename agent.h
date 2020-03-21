//
// Created by 周梓康 on 2020/3/2.
//

#ifndef NIM_AGENT_H_
#define NIM_AGENT_H_

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "action.h"
#include "state.h"

class Game;

class Agent {
  friend void swap(Game &, Game &);
  friend class Game;

 public:
  using Reward = double;
  Agent() = default;
  Agent(const Agent &) = delete;
  Agent(Agent &&agent) noexcept { MoveGames(&agent); }
  Agent &operator=(const Agent &) = delete;
  Agent &operator=(Agent &&) noexcept;
  virtual ~Agent() { RemoveFromGames(); }
  State &current_state() { return current_state_; }
  const State &current_state() const { return current_state_; }
  template<typename T>
  void set_current_state(T &&current_state) {
    current_state_ = std::forward<T>(current_state);
  }
  std::unordered_set<Game *> &games() { return games_; }
  const std::unordered_set<Game *> &games() const { return games_; }
  virtual Action Policy(const State &state) = 0;
  void Reset();
  virtual void Update(const State &current_state,
                      const State &next_state,
                      Reward reward) {}

 protected:
  State current_state_;
  State next_state_greedy_;
  std::unordered_set<Game *> games_;
  Action SampleAction(const std::vector<Action> &);

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
  Action Policy(const State &state) override;
};

class HumanAgent : public Agent {
 public:
  explicit HumanAgent(std::istream &is = std::cin, std::ostream &os = std::cout)
      : Agent(), is_(is), os_(os) {}

  HumanAgent(const HumanAgent &) = delete;
  HumanAgent(HumanAgent &&) = default;
  HumanAgent &operator=(const HumanAgent &) = delete;
  HumanAgent &operator=(HumanAgent &&) = delete;
  Action Policy(const State &state) override;

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
  Action Policy(const State &state) override;
};

class QLearningAgent : public Agent {
 public:
  explicit QLearningAgent(double alpha = 1.0,
                          double gamma = 1.0,
                          double epsilon = 0.5,
                          double decay_epsilon = 1.0)
      : alpha_(alpha),
        gamma_(gamma),
        epsilon_(epsilon),
        decay_epsilon_(decay_epsilon) {}
  QLearningAgent(const QLearningAgent &) = delete;
  QLearningAgent(QLearningAgent &&) = default;
  QLearningAgent &operator=(const QLearningAgent &) = delete;
  QLearningAgent &operator=(QLearningAgent &&) = default;
  double alpha() { return alpha_; }
  const double alpha() const { return alpha_; }
  double decay_epsilon() { return decay_epsilon_; }
  const double decay_epsilon() const { return decay_epsilon_; }
  double epsilon() { return epsilon_; }
  const double epsilon() const { return epsilon_; }
  double gamma() { return gamma_; }
  const double gamma() const { return gamma_; }
  double ConvergenceRate();
  void InitQValues(const State &);
  Action Policy(const State &state) override;
  std::unordered_map<State, Reward> &q_values() { return q_values_; }
  const std::unordered_map<State,
                           Reward> &q_values() const { return q_values_; }
  void set_alpha(double alpha) { alpha_ = alpha; }
  void set_decay_epsilon(double decay_epsilon) {
    decay_epsilon_ = decay_epsilon;
  }
  void set_epsilon(double epsilon) { epsilon_ = epsilon; }
  void set_gamma(double gamma) { gamma_ = gamma; }
  template<typename T>
  void set_q_values_(T &&q_values) { q_values_ = std::forward<T>(q_values); }
  void Update(const State &current_state,
              const State &next_state,
              Reward reward) override;

 private:
  double alpha_;
  double gamma_;
  double epsilon_;
  double decay_epsilon_;
  std::unordered_map<State, Reward> q_values_;
  std::default_random_engine generator_;
  std::uniform_real_distribution<> epsilon_distribution_{0, 1};
  std::uniform_real_distribution<> q_value_distribution_{-1, 1};
  void InitQValuesImpl(const State &state, int pile_id);
};

std::ostream &operator<<(std::ostream &,
                         const std::unordered_map<State, Agent::Reward> &);

#endif  // NIM_AGENT_H_
