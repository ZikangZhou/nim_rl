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

#ifndef NIM_RL_AGENT_MONTE_CARLO_AGENT_H_
#define NIM_RL_AGENT_MONTE_CARLO_AGENT_H_

#include "nim_rl/agent/rl_agent.h"
#include "nim_rl/exploration/exploration.h"

namespace nim_rl {

enum class ImportanceSampling {
  kWeighted,
  kNormal,
};

class MonteCarloAgent : public RLAgent {
 public:
  explicit
  MonteCarloAgent(double gamma = kDefaultGamma) : gamma_(gamma) {}
  MonteCarloAgent(const MonteCarloAgent &) = default;
  MonteCarloAgent(MonteCarloAgent &&) = default;
  MonteCarloAgent &operator=(const MonteCarloAgent &) = default;
  MonteCarloAgent &operator=(MonteCarloAgent &&) = default;
  ~MonteCarloAgent() override = default;
  std::shared_ptr<Agent> Clone() const override {
    return std::shared_ptr<Agent>(new MonteCarloAgent(*this));
  }
  double GetGamma() const { return gamma_; }
  void Initialize(const std::vector<State> &) override;
  Action PolicyImpl(const std::vector<Action> &/*legal_actions*/,
                    const std::vector<Action> &greedy_actions) override {
    return SampleAction(greedy_actions);
  }
  void Reset() override;
  void SetGamma(double gamma) { gamma_ = gamma; }
  Action Step(Game *, bool is_evaluation) override;
  void Update(const State &update_state, const State &current_state,
              Reward reward) override;

 protected:
  double gamma_;
  std::vector<TimeStep> trajectory_;
  std::unordered_map<State, double> cumulative_sums_;
};

class ESMonteCarloAgent : public MonteCarloAgent {
 public:
  explicit ESMonteCarloAgent(double gamma = kDefaultGamma)
      : MonteCarloAgent(gamma) {}
  ESMonteCarloAgent(const ESMonteCarloAgent &) = default;
  ESMonteCarloAgent(ESMonteCarloAgent &&) = default;
  ESMonteCarloAgent &operator=(const ESMonteCarloAgent &) = default;
  ESMonteCarloAgent &operator=(ESMonteCarloAgent &&) = default;
  ~ESMonteCarloAgent() override = default;
  std::shared_ptr<Agent> Clone() const override {
    return std::shared_ptr<Agent>(new ESMonteCarloAgent(*this));
  }
  Action Step(Game *, bool is_evaluation) override;
};

class OnPolicyMonteCarloAgent : public MonteCarloAgent {
 public:
  explicit
  OnPolicyMonteCarloAgent(double gamma = kDefaultGamma,
                          const Exploration &exploration = EpsilonGreedy())
      : MonteCarloAgent(gamma), exploration_(exploration.Clone()) {}
  OnPolicyMonteCarloAgent(const OnPolicyMonteCarloAgent &) = default;
  OnPolicyMonteCarloAgent(OnPolicyMonteCarloAgent &&) = default;
  OnPolicyMonteCarloAgent &operator=(const OnPolicyMonteCarloAgent &) = default;
  OnPolicyMonteCarloAgent &operator=(OnPolicyMonteCarloAgent &&) = default;
  ~OnPolicyMonteCarloAgent() override = default;
  std::shared_ptr<Agent> Clone() const override {
    return std::shared_ptr<Agent>(new OnPolicyMonteCarloAgent(*this));
  }
  Action PolicyImpl(const std::vector<Action> &legal_actions,
                    const std::vector<Action> &greedy_actions) override {
    return exploration_->Explore(legal_actions, greedy_actions);
  }
  void UpdateExploration(int episode) override {
    exploration_->Update(episode);
  }

 private:
  std::shared_ptr<Exploration> exploration_;
};

class OffPolicyMonteCarloAgent : public MonteCarloAgent {
 public:
  explicit OffPolicyMonteCarloAgent(
      double gamma = kDefaultGamma,
      ImportanceSampling importance_sampling = ImportanceSampling::kWeighted,
      double epsilon = kDefaultEpsilon,
      double epsilon_decay_factor = kDefaultEpsilonDecayFactor,
      double min_epsilon = kDefaultMinEpsilon)
      : MonteCarloAgent(gamma),
        importance_sampling_(importance_sampling),
        epsilon_greedy_(epsilon, epsilon_decay_factor, min_epsilon) {}
  OffPolicyMonteCarloAgent(const OffPolicyMonteCarloAgent &) = default;
  OffPolicyMonteCarloAgent(OffPolicyMonteCarloAgent &&) = default;
  OffPolicyMonteCarloAgent &
  operator=(const OffPolicyMonteCarloAgent &) = default;
  OffPolicyMonteCarloAgent &operator=(OffPolicyMonteCarloAgent &&) = default;
  ~OffPolicyMonteCarloAgent() override = default;
  std::shared_ptr<Agent> Clone() const override {
    return std::shared_ptr<Agent>(new OffPolicyMonteCarloAgent(*this));
  }
  Action PolicyImpl(const std::vector<Action> &legal_actions,
                    const std::vector<Action> &greedy_actions) override {
    return epsilon_greedy_.Explore(legal_actions, greedy_actions);
  }
  void Update(const State &update_state, const State &current_state,
              Reward reward) override;
  void UpdateExploration(int episode) override {
    epsilon_greedy_.Update(episode);
  }

 private:
  ImportanceSampling importance_sampling_;
  EpsilonGreedy epsilon_greedy_;
};

}  // namespace nim_rl

#endif  // NIM_RL_AGENT_MONTE_CARLO_AGENT_H_
