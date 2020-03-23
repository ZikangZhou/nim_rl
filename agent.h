//
// Created by 周梓康 on 2020/3/2.
//

#ifndef NIM_AGENT_H_
#define NIM_AGENT_H_

#include <algorithm>
#include <iomanip>
#include <iostream>
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
  State &GetCurrentState() { return current_state_; }
  const State &GetCurrentState() const { return current_state_; }
  std::unordered_set<Game *> &GetGames() { return games_; }
  const std::unordered_set<Game *> &GetGames() const { return games_; }
  virtual Action Policy(const State &) = 0;
  virtual void Reset() { current_state_ = State(); }
  template<typename T>
  void SetCurrentState(T &&current_state) {
    current_state_ = std::forward<T>(current_state);
  }
  virtual void Update(const State &current_state,
                      const State &next_state,
                      Reward reward) { current_state_ = next_state; }

 protected:
  State current_state_;
  std::unordered_set<Game *> games_;
  Action SampleAction(const std::vector<Action> &);

 private:
  std::mt19937 rng_{std::random_device{}()};
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
  Action Policy(const State &) override;
};

class HumanAgent : public Agent {
 public:
  explicit HumanAgent(std::istream &is = std::cin, std::ostream &os = std::cout)
      : Agent(), is_(is), os_(os) {}
  HumanAgent(const HumanAgent &) = delete;
  HumanAgent(HumanAgent &&) = default;
  HumanAgent &operator=(const HumanAgent &) = delete;
  HumanAgent &operator=(HumanAgent &&) = delete;
  Action Policy(const State &) override;

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
  Action Policy(const State &) override;
};

class QLearningAgent : public Agent {
 public:
  explicit QLearningAgent(double alpha = 1.0,
                          double gamma = 1.0,
                          double epsilon = 0.5,
                          double epsilon_decay_factor = 1.0)
      : alpha_(alpha),
        gamma_(gamma),
        epsilon_(epsilon),
        epsilon_decay_factor_(epsilon_decay_factor) {}
  QLearningAgent(const QLearningAgent &) = delete;
  QLearningAgent(QLearningAgent &&) = default;
  QLearningAgent &operator=(const QLearningAgent &) = delete;
  QLearningAgent &operator=(QLearningAgent &&) = default;
  double GetAlpha() const { return alpha_; }
  double GetEpsilon() const { return epsilon_; }
  double GetEpsilonDecayFactor() const { return epsilon_decay_factor_; }
  double GetGamma() const { return gamma_; }
  State &GetNextStateGreedy() { return next_state_greedy_; }
  const State &GetNextStateGreedy() const { return next_state_greedy_; }
  std::unordered_map<State, Reward> &GetQValues() { return q_values_; }
  const std::unordered_map<State,
                           Reward> &GetQValues() const { return q_values_; }
  void InitializeQValues(const State &);
  double OptimalActionsRatio();
  Action Policy(const State &) override;
  void Reset() override;
  void SetAlpha(double alpha) { alpha_ = alpha; }
  void SetEpsilon(double epsilon) { epsilon_ = epsilon; }
  void SetEpsilonDecayFactor(double decay_epsilon) {
    epsilon_decay_factor_ = decay_epsilon;
  }
  void SetGamma(double gamma) { gamma_ = gamma; }
  template<typename T>
  void SetNextStateGreedy(T &&next_state_greedy) {
    next_state_greedy_ = std::forward<T>(next_state_greedy);
  }
  template<typename T>
  void SetQValues(T &&q_values) { q_values_ = std::forward<T>(q_values); }
  void Update(const State &current_state,
              const State &next_state,
              Reward reward) override;
  void UpdateEpsilon() {
    epsilon_ = std::max(0.1, epsilon_ * epsilon_decay_factor_);
  }

 private:
  double alpha_;
  double gamma_;
  double epsilon_;
  double epsilon_decay_factor_;
  State next_state_greedy_;
  std::unordered_map<State, Reward> q_values_;
  std::mt19937 rng_{std::random_device{}()};
  std::uniform_real_distribution<> dist_epsilon_{0, 1};
  std::uniform_real_distribution<> dist_q_value_{-1, 1};
  void DoInitializeQValues(const State &state, int pile_id);
};

std::ostream &operator<<(std::ostream &,
                         const std::unordered_map<State, Agent::Reward> &);

#endif  // NIM_AGENT_H_
