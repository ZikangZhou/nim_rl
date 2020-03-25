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

namespace std {
template<>
struct hash<std::pair<State, Action>> {
  std::size_t operator()(const std::pair<State, Action> &state_action) const {
    std::size_t seed = 0;
    seed ^= std::hash<State>()(state_action.first) + 0x9e3779b9
        + (seed << 6) + (seed >> 2);
    seed ^= std::hash<Action>()(state_action.second) + 0x9e3779b9
        + (seed << 6) + (seed >> 2);
    return seed;
  }
};
}  // namespace std

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

class ValueIterationAgent : public Agent {
 public:
  using StateAction = std::pair<State, Action>;
  using StateProb = std::pair<State, double>;
  explicit ValueIterationAgent(double threshold = 1e-4)
      : Agent(), threshold_(threshold) {}
  ValueIterationAgent(const ValueIterationAgent &) = delete;
  ValueIterationAgent(ValueIterationAgent &&) = default;
  ValueIterationAgent &operator=(const ValueIterationAgent &) = delete;
  ValueIterationAgent &operator=(ValueIterationAgent &&) = default;
  double GetThreshold() const { return threshold_; }
  std::unordered_map<StateAction,
                     std::vector<StateProb>> GetTransitions() const {
    return transitions_;
  }
  std::unordered_map<State, Reward> GetValues() const { return values_; }
  void InitializeMaps(const std::vector<State> &);
  Action Policy(const State &) override;
  void SetThreshold(double threshold) { threshold_ = threshold; }
  template<typename T>
  void SetTransitions(T &&transitions) {
    transitions_ = std::forward<T>(transitions);
  }
  template<typename T>
  void SetValues(T &&values) { values_ = std::forward<T>(values); }
  void Train(const State &);

 private:
  std::unordered_map<State, Reward> values_;
  std::unordered_map<StateAction, std::vector<StateProb>> transitions_;
  double threshold_;
};

class TDAgent : public Agent {
 public:
  explicit TDAgent(double alpha = 1.0,
                   double gamma = 1.0,
                   double epsilon = 0.5,
                   double epsilon_decay_factor = 1.0)
      : alpha_(alpha),
        gamma_(gamma),
        epsilon_(epsilon),
        epsilon_decay_factor_(epsilon_decay_factor) {}
  TDAgent(const TDAgent &) = delete;
  TDAgent(TDAgent &&) = default;
  TDAgent &operator=(const TDAgent &) = delete;
  TDAgent &operator=(TDAgent &&) = default;
  double GetAlpha() const { return alpha_; }
  double GetEpsilon() const { return epsilon_; }
  double GetEpsilonDecayFactor() const { return epsilon_decay_factor_; }
  double GetGamma() const { return gamma_; }
  virtual std::unordered_map<State, Reward> GetValues() const {
    return values_;
  }
  virtual void InitializeValues(const State &);
  double OptimalActionsRatio();
  Action Policy(const State &) override;
  void Reset() override;
  void SetAlpha(double alpha) { alpha_ = alpha; }
  void SetEpsilon(double epsilon) { epsilon_ = epsilon; }
  void SetEpsilonDecayFactor(double decay_epsilon) {
    epsilon_decay_factor_ = decay_epsilon;
  }
  void SetGamma(double gamma) { gamma_ = gamma; }
  virtual void SetValues(const std::unordered_map<State, Reward> &values) {
    values_ = values;
  }
  virtual void SetValues(std::unordered_map<State, Reward> &&values) {
    values_ = std::move(values);
  }
  void UpdateEpsilon() {
    epsilon_ = std::max(0.01, epsilon_ * epsilon_decay_factor_);
  }

 protected:
  double alpha_;
  double gamma_;
  double epsilon_;
  double epsilon_decay_factor_;
  std::unordered_map<State, Reward> values_;
  Reward greedy_value_ = 0.0;
  std::vector<Action> legal_actions_;
  unsigned long num_legal_actions_ = 0;
  unsigned long num_greedy_actions_ = 0;
  Action EpsilonGreedy(const std::vector<Action> &legal_actions,
                       const std::vector<Action> &greedy_actions) {
    return (dist_epsilon_(rng_) < epsilon_) ? SampleAction(legal_actions)
                                            : SampleAction(greedy_actions);
  }

 private:
  std::uniform_real_distribution<> dist_epsilon_{0, 1};
  std::uniform_real_distribution<> dist_value_{-1, 1};
  std::mt19937 rng_{std::random_device{}()};
  void DoInitializeValues(const State &state, int pile_id);
};

class QLearningAgent : public TDAgent {
 public:
  explicit QLearningAgent(double alpha = 1.0,
                          double gamma = 1.0,
                          double epsilon = 0.5,
                          double epsilon_decay_factor = 1.0)
      : TDAgent(alpha, gamma, epsilon, epsilon_decay_factor) {}
  QLearningAgent(const QLearningAgent &) = delete;
  QLearningAgent(QLearningAgent &&) = default;
  QLearningAgent &operator=(const QLearningAgent &) = delete;
  QLearningAgent &operator=(QLearningAgent &&) = default;
  void Update(const State &current_state,
              const State &next_state,
              Reward reward) override;
};

class SarsaAgent : public TDAgent {
 public:
  explicit SarsaAgent(double alpha = 1.0,
                      double gamma = 1.0,
                      double epsilon = 0.5,
                      double epsilon_decay_factor = 1.0)
      : TDAgent(alpha, gamma, epsilon, epsilon_decay_factor) {}
  SarsaAgent(const SarsaAgent &) = delete;
  SarsaAgent(SarsaAgent &&) = default;
  SarsaAgent &operator=(const SarsaAgent &) = delete;
  SarsaAgent &operator=(SarsaAgent &&) = default;
  void Update(const State &current_state,
              const State &next_state,
              Reward reward) override;
};

class ExpectedSarsaAgent : public TDAgent {
 public:
  explicit ExpectedSarsaAgent(double alpha = 1.0,
                              double gamma = 1.0,
                              double epsilon = 0.5,
                              double epsilon_decay_factor = 1.0)
      : TDAgent(alpha, gamma, epsilon, epsilon_decay_factor) {}
  ExpectedSarsaAgent(const ExpectedSarsaAgent &) = delete;
  ExpectedSarsaAgent(ExpectedSarsaAgent &&) = default;
  ExpectedSarsaAgent &operator=(const ExpectedSarsaAgent &) = delete;
  ExpectedSarsaAgent &operator=(ExpectedSarsaAgent &&) = default;
  std::vector<State> &GetNextStates() { return next_states_; }
  const std::vector<State> &GetNextStates() const { return next_states_; }
  Action Policy(const State &) override;
  void Reset() override;
  template<typename T>
  void SetNextStates(T &&next_states) {
    next_states_ = std::forward<T>(next_states);
  }
  void Update(const State &current_state,
              const State &next_state,
              Reward reward) override;

 private:
  std::vector<State> next_states_;
};

class DoubleLearningAgent : public TDAgent {
 public:
  explicit DoubleLearningAgent(double alpha = 1.0,
                               double gamma = 1.0,
                               double epsilon = 0.5,
                               double epsilon_decay_factor = 1.0)
      : TDAgent(alpha, gamma, epsilon, epsilon_decay_factor) {}
  DoubleLearningAgent(const DoubleLearningAgent &) = delete;
  DoubleLearningAgent(DoubleLearningAgent &&) = default;
  DoubleLearningAgent &operator=(const DoubleLearningAgent &) = delete;
  DoubleLearningAgent &operator=(DoubleLearningAgent &&) = default;
  std::unordered_map<State, Reward> GetValues() const override;
  void InitializeValues(const State &) override;
  Action Policy(const State &) override;
  void Reset() override;
  void SetValues(const std::unordered_map<State, Reward> &values) override {
    values_ = values_2_ = values;
  }
  void SetValues(std::unordered_map<State, Reward> &&values) override {
    values_ = values_2_ = std::move(values);
  }
  void Update(const State &current_state,
              const State &next_state,
              Reward reward) override;

 protected:
  std::unordered_map<State, Reward> values_2_;
  double prob_ = 0.0;
  virtual void DoUpdate(const State &current_state,
                        const State &next_state,
                        Reward reward,
                        std::unordered_map<State, Reward> *values) = 0;

 private:
  std::uniform_real_distribution<> dist_{0, 1};
  std::mt19937 rng_{std::random_device{}()};
};

class DoubleQLearningAgent : public DoubleLearningAgent {
 public:
  explicit DoubleQLearningAgent(double alpha = 1.0,
                                double gamma = 1.0,
                                double epsilon = 0.5,
                                double epsilon_decay_factor = 1.0)
      : DoubleLearningAgent(alpha, gamma, epsilon, epsilon_decay_factor) {}
  DoubleQLearningAgent(const DoubleQLearningAgent &) = delete;
  DoubleQLearningAgent(DoubleQLearningAgent &&) = default;
  DoubleQLearningAgent &operator=(const DoubleQLearningAgent &) = delete;
  DoubleQLearningAgent &operator=(DoubleQLearningAgent &&) = default;

 private:
  void DoUpdate(const State &current_state,
                const State &next_state,
                Reward reward,
                std::unordered_map<State, Reward> *values) override;
};

class DoubleSarsaAgent : public DoubleLearningAgent {
 public:
  explicit DoubleSarsaAgent(double alpha = 1.0,
                            double gamma = 1.0,
                            double epsilon = 0.5,
                            double epsilon_decay_factor = 1.0)
      : DoubleLearningAgent(alpha, gamma, epsilon, epsilon_decay_factor) {}
  DoubleSarsaAgent(const DoubleSarsaAgent &) = delete;
  DoubleSarsaAgent(DoubleSarsaAgent &&) = default;
  DoubleSarsaAgent &operator=(const DoubleSarsaAgent &) = delete;
  DoubleSarsaAgent &operator=(DoubleSarsaAgent &&) = default;

 private:
  void DoUpdate(const State &current_state,
                const State &next_state,
                Reward reward,
                std::unordered_map<State, Reward> *values) override;
};

class DoubleExpectedSarsaAgent : public DoubleLearningAgent {
 public:
  explicit DoubleExpectedSarsaAgent(double alpha = 1.0,
                                    double gamma = 1.0,
                                    double epsilon = 0.5,
                                    double epsilon_decay_factor = 1.0)
      : DoubleLearningAgent(alpha, gamma, epsilon, epsilon_decay_factor) {}
  DoubleExpectedSarsaAgent(const DoubleExpectedSarsaAgent &) = delete;
  DoubleExpectedSarsaAgent(DoubleExpectedSarsaAgent &&) = default;
  DoubleExpectedSarsaAgent &
  operator=(const DoubleExpectedSarsaAgent &) = delete;
  DoubleExpectedSarsaAgent &operator=(DoubleExpectedSarsaAgent &&) = default;
  std::vector<State> &GetNextStates() { return next_states_; }
  const std::vector<State> &GetNextStates() const { return next_states_; }
  Action Policy(const State &) override;
  void Reset() override;
  template<typename T>
  void SetNextStates(T &&next_states) {
    next_states_ = std::forward<T>(next_states);
  }

 private:
  std::vector<State> next_states_;
  void DoUpdate(const State &current_state,
                const State &next_state,
                Reward reward,
                std::unordered_map<State, Reward> *values) override;
};

std::ostream &operator<<(std::ostream &,
                         const std::unordered_map<State, Agent::Reward> &);

#endif  // NIM_AGENT_H_
