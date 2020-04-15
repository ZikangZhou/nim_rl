// Copyright 2020 Zhou Zikang. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NIM_RL_AGENT_TD_AGENT_H_
#define NIM_RL_AGENT_TD_AGENT_H_

#include "nim_rl/agent/rl_agent.h"
#include "nim_rl/exploration/exploration.h"

namespace nim_rl {

class TDAgent : public RLAgent {
 public:
  explicit TDAgent(double alpha = kDefaultAlpha,
                   double gamma = kDefaultGamma,
                   double epsilon = kDefaultEpsilon,
                   double epsilon_decay_factor = kDefaultEpsilonDecayFactor,
                   double min_epsilon = kDefaultMinEpsilon)
      : alpha_(alpha),
        gamma_(gamma),
        epsilon_greedy_(epsilon, epsilon_decay_factor, min_epsilon) {}
  TDAgent(const TDAgent &) = default;
  TDAgent(TDAgent &&) = default;
  TDAgent &operator=(const TDAgent &) = default;
  TDAgent &operator=(TDAgent &&) = default;
  ~TDAgent() override = default;
  std::shared_ptr<Agent> Clone() const & override {
    return std::shared_ptr<TDAgent>(new TDAgent(*this));
  }
  std::shared_ptr<Agent> Clone() && override {
    return std::shared_ptr<Agent>(new TDAgent(std::move(*this)));
  }
  double GetAlpha() const { return alpha_; }
  double GetGamma() const { return gamma_; }
  void SetAlpha(double alpha) { alpha_ = alpha; }
  void SetGamma(double gamma) { gamma_ = gamma; }
  Action Step(Game *, bool is_evaluation) override;
  void UpdateExploration() override { epsilon_greedy_.Update(); }

 protected:
  double alpha_;
  double gamma_;
  EpsilonGreedy epsilon_greedy_;
  Action PolicyImpl(const std::vector<Action> &legal_actions,
                    const std::vector<Action> &greedy_actions) override {
    return epsilon_greedy_.Explore(legal_actions, greedy_actions);
  }
};

class QLearningAgent : public TDAgent {
 public:
  explicit
  QLearningAgent(double alpha = kDefaultAlpha,
                 double gamma = kDefaultGamma,
                 double epsilon = kDefaultEpsilon,
                 double epsilon_decay_factor = kDefaultEpsilonDecayFactor,
                 double min_epsilon = kDefaultMinEpsilon)
      : TDAgent(alpha, gamma, epsilon, epsilon_decay_factor, min_epsilon) {}
  QLearningAgent(const QLearningAgent &) = default;
  QLearningAgent(QLearningAgent &&) = default;
  QLearningAgent &operator=(const QLearningAgent &) = default;
  QLearningAgent &operator=(QLearningAgent &&) = default;
  ~QLearningAgent() override = default;
  std::shared_ptr<Agent> Clone() const & override {
    return std::shared_ptr<Agent>(new QLearningAgent(*this));
  }
  std::shared_ptr<Agent> Clone() && override {
    return std::shared_ptr<Agent>(new QLearningAgent(std::move(*this)));
  }

 protected:
  void Update(const State &update_state, const State &current_state,
              Reward reward) override;
};

class SarsaAgent : public TDAgent {
 public:
  explicit SarsaAgent(double alpha = kDefaultAlpha,
                      double gamma = kDefaultGamma,
                      double epsilon = kDefaultEpsilon,
                      double epsilon_decay_factor = kDefaultEpsilonDecayFactor,
                      double min_epsilon = kDefaultMinEpsilon)
      : TDAgent(alpha, gamma, epsilon, epsilon_decay_factor, min_epsilon) {}
  SarsaAgent(const SarsaAgent &) = default;
  SarsaAgent(SarsaAgent &&) = default;
  SarsaAgent &operator=(const SarsaAgent &) = default;
  SarsaAgent &operator=(SarsaAgent &&) = default;
  ~SarsaAgent() override = default;
  std::shared_ptr<Agent> Clone() const & override {
    return std::shared_ptr<Agent>(new SarsaAgent(*this));
  }
  std::shared_ptr<Agent> Clone() && override {
    return std::shared_ptr<Agent>(new SarsaAgent(std::move(*this)));
  }

 protected:
  void Update(const State &update_state, const State &current_state,
              Reward reward) override;
};

class ExpectedSarsaAgent : public TDAgent {
 public:
  explicit
  ExpectedSarsaAgent(double alpha = kDefaultAlpha,
                     double gamma = kDefaultGamma,
                     double epsilon = kDefaultEpsilon,
                     double epsilon_decay_factor = kDefaultEpsilonDecayFactor,
                     double min_epsilon = kDefaultMinEpsilon)
      : TDAgent(alpha, gamma, epsilon, epsilon_decay_factor, min_epsilon) {}
  ExpectedSarsaAgent(const ExpectedSarsaAgent &) = default;
  ExpectedSarsaAgent(ExpectedSarsaAgent &&) = default;
  ExpectedSarsaAgent &operator=(const ExpectedSarsaAgent &) = default;
  ExpectedSarsaAgent &operator=(ExpectedSarsaAgent &&) = default;
  ~ExpectedSarsaAgent() override = default;
  std::shared_ptr<Agent> Clone() const & override {
    return std::shared_ptr<Agent>(new ExpectedSarsaAgent(*this));
  }
  std::shared_ptr<Agent> Clone() && override {
    return std::shared_ptr<Agent>(new ExpectedSarsaAgent(std::move(*this)));
  }
  void Reset() override;

 protected:
  void Update(const State &update_state, const State &current_state,
              Reward reward) override;

 private:
  std::vector<State> next_states_;
  Action Policy(const State &, bool is_evaluation) override;
  template<typename T>
  void SetNextStates(T &&next_states) {
    next_states_ = std::forward<T>(next_states);
  }
};

class DoubleLearningAgent : public TDAgent {
 public:
  explicit
  DoubleLearningAgent(double alpha = kDefaultAlpha,
                      double gamma = kDefaultGamma,
                      double epsilon = kDefaultEpsilon,
                      double epsilon_decay_factor = kDefaultEpsilonDecayFactor,
                      double min_epsilon = kDefaultMinEpsilon)
      : TDAgent(alpha, gamma, epsilon, epsilon_decay_factor, min_epsilon) {}
  DoubleLearningAgent(const DoubleLearningAgent &) = default;
  DoubleLearningAgent(DoubleLearningAgent &&) = default;
  DoubleLearningAgent &operator=(const DoubleLearningAgent &) = default;
  DoubleLearningAgent &operator=(DoubleLearningAgent &&) = default;
  ~DoubleLearningAgent() override = default;
  std::unordered_map<State, Reward> GetValues() const override;
  void Initialize(const std::vector<State> &) override;
  void Reset() override;
  void SetValues(const std::unordered_map<State, Reward> &values) override {
    values_ = values_2_ = values;
  }
  void SetValues(std::unordered_map<State, Reward> &&values) override {
    values_ = values_2_ = std::move(values);
  }

 protected:
  std::unordered_map<State, Reward> values_2_;
  bool flag_ = false;
  virtual void DoUpdate(const State &update_state, const State &current_state,
                        Reward reward,
                        std::unordered_map<State, Reward> *values) = 0;
  Action Policy(const State &, bool is_evaluation) override;
  void Update(const State &update_state, const State &current_state,
              Reward reward) override;

 private:
  std::bernoulli_distribution dist_flag_{};
  std::mt19937 rng_{std::random_device{}()};
};

class DoubleQLearningAgent : public DoubleLearningAgent {
 public:
  explicit
  DoubleQLearningAgent(double alpha = kDefaultAlpha,
                       double gamma = kDefaultGamma,
                       double epsilon = kDefaultEpsilon,
                       double epsilon_decay_factor = kDefaultEpsilonDecayFactor,
                       double min_epsilon = kDefaultMinEpsilon)
      : DoubleLearningAgent(alpha, gamma, epsilon, epsilon_decay_factor,
                            min_epsilon) {}
  DoubleQLearningAgent(const DoubleQLearningAgent &) = default;
  DoubleQLearningAgent(DoubleQLearningAgent &&) = default;
  DoubleQLearningAgent &operator=(const DoubleQLearningAgent &) = default;
  DoubleQLearningAgent &operator=(DoubleQLearningAgent &&) = default;
  ~DoubleQLearningAgent() override = default;
  std::shared_ptr<Agent> Clone() const & override {
    return std::shared_ptr<Agent>(new DoubleQLearningAgent(*this));
  }
  std::shared_ptr<Agent> Clone() && override {
    return std::shared_ptr<Agent>(new DoubleQLearningAgent(std::move(*this)));
  }

 private:
  void DoUpdate(const State &update_state,
                const State &current_state,
                Reward reward,
                std::unordered_map<State, Reward> *values) override;
};

class DoubleSarsaAgent : public DoubleLearningAgent {
 public:
  explicit
  DoubleSarsaAgent(double alpha = kDefaultAlpha,
                   double gamma = kDefaultGamma,
                   double epsilon = kDefaultEpsilon,
                   double epsilon_decay_factor = kDefaultEpsilonDecayFactor,
                   double min_epsilon = kDefaultMinEpsilon)
      : DoubleLearningAgent(alpha, gamma, epsilon, epsilon_decay_factor,
                            min_epsilon) {}
  DoubleSarsaAgent(const DoubleSarsaAgent &) = default;
  DoubleSarsaAgent(DoubleSarsaAgent &&) = default;
  DoubleSarsaAgent &operator=(const DoubleSarsaAgent &) = default;
  DoubleSarsaAgent &operator=(DoubleSarsaAgent &&) = default;
  ~DoubleSarsaAgent() override = default;
  std::shared_ptr<Agent> Clone() const & override {
    return std::shared_ptr<Agent>(new DoubleSarsaAgent(*this));
  }
  std::shared_ptr<Agent> Clone() && override {
    return std::shared_ptr<Agent>(new DoubleSarsaAgent(std::move(*this)));
  }

 private:
  void DoUpdate(const State &update_state,
                const State &current_state,
                Reward reward,
                std::unordered_map<State, Reward> *values) override;
};

class DoubleExpectedSarsaAgent : public DoubleLearningAgent {
 public:
  explicit DoubleExpectedSarsaAgent(
      double alpha = kDefaultAlpha,
      double gamma = kDefaultGamma,
      double epsilon = kDefaultEpsilon,
      double epsilon_decay_factor = kDefaultEpsilonDecayFactor,
      double min_epsilon = kDefaultMinEpsilon)
      : DoubleLearningAgent(alpha, gamma, epsilon, epsilon_decay_factor,
                            min_epsilon) {}
  DoubleExpectedSarsaAgent(const DoubleExpectedSarsaAgent &) = default;
  DoubleExpectedSarsaAgent(DoubleExpectedSarsaAgent &&) = default;
  DoubleExpectedSarsaAgent &
  operator=(const DoubleExpectedSarsaAgent &) = default;
  DoubleExpectedSarsaAgent &operator=(DoubleExpectedSarsaAgent &&) = default;
  ~DoubleExpectedSarsaAgent() override = default;
  std::shared_ptr<Agent> Clone() const & override {
    return std::shared_ptr<Agent>(new DoubleExpectedSarsaAgent(*this));
  }
  std::shared_ptr<Agent> Clone() && override {
    return std::shared_ptr<Agent>(
        new DoubleExpectedSarsaAgent(std::move(*this)));
  }
  void Reset() override;

 private:
  std::vector<State> next_states_;
  void DoUpdate(const State &update_state,
                const State &current_state,
                Reward reward,
                std::unordered_map<State, Reward> *values) override;
  Action Policy(const State &, bool is_evaluation) override;
  template<typename T>
  void SetNextStates(T &&next_states) {
    next_states_ = std::forward<T>(next_states);
  }
};

}  // namespace nim_rl

#endif  // NIM_RL_AGENT_TD_AGENT_H_
