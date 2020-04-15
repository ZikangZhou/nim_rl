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

#ifndef NIM_RL_EXPLORATION_EXPLORATION_H_
#define NIM_RL_EXPLORATION_EXPLORATION_H_

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

#include "nim_rl/action/action.h"
#include "nim_rl/agent/agent.h"

namespace nim_rl {

constexpr double kDefaultEpsilon = 1.0;
constexpr double kDefaultEpsilonDecayFactor = 0.9;
constexpr double kDefaultMinEpsilon = 0.01;

class Exploration {
 public:
  Exploration() = default;
  virtual std::shared_ptr<Exploration> Clone() const & = 0;
  virtual std::shared_ptr<Exploration> Clone() && = 0;
  virtual Action Explore(const std::vector<nim_rl::Action> &legal_actions,
                         const std::vector<nim_rl::Action> &greedy_actions) = 0;
  virtual void Update() = 0;
};

class EpsilonGreedy : public Exploration {
 public:
  explicit
  EpsilonGreedy(double epsilon = kDefaultEpsilon,
                double epsilon_decay_factor = kDefaultEpsilonDecayFactor,
                double min_epsilon = kDefaultMinEpsilon)
      : epsilon_(epsilon),
        epsilon_decay_factor_(epsilon_decay_factor),
        min_epsilon_(min_epsilon) {}
  std::shared_ptr<Exploration> Clone() const & override {
    return std::shared_ptr<Exploration>(new EpsilonGreedy(*this));
  }
  std::shared_ptr<Exploration> Clone() && override {
    return std::shared_ptr<Exploration>(new EpsilonGreedy(std::move(*this)));
  }
  Action Explore(const std::vector<Action> &legal_actions,
                 const std::vector<Action> &greedy_actions) override {
    return (dist_epsilon_(rng_) < epsilon_) ? SampleAction(legal_actions)
                                            : SampleAction(greedy_actions);
  }
  double GetEpsilon() const { return epsilon_; }
  double GetEpsilonDecayFactor() const { return epsilon_decay_factor_; }
  double GetMinEpsilon() const { return min_epsilon_; }
  void SetEpsilon(double epsilon) { epsilon_ = epsilon; }
  void SetEpsilonDecayFactor(double decay_epsilon) {
    epsilon_decay_factor_ = decay_epsilon;
  }
  void SetMinEpsilon(double min_epsilon) { min_epsilon_ = min_epsilon; }
  void Update() override {
    epsilon_ = std::max(min_epsilon_, epsilon_ * epsilon_decay_factor_);
  }

 protected:
  double epsilon_;
  double epsilon_decay_factor_;
  double min_epsilon_;

 private:
  std::uniform_real_distribution<> dist_epsilon_{0, 1};
  std::mt19937 rng_{std::random_device{}()};
};

}  // namespace nim_rl

#endif  // NIM_RL_EXPLORATION_EXPLORATION_H_
