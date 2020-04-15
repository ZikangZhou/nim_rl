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

#ifndef NIM_RL_AGENT_DP_AGENT_H_
#define NIM_RL_AGENT_DP_AGENT_H_

#include "nim_rl/agent/rl_agent.h"

namespace nim_rl {

constexpr double kDefaultThreshold = 1e-4;

class DPAgent : public RLAgent {
 public:
  explicit DPAgent(double gamma = kDefaultGamma,
                   double threshold = kDefaultThreshold)
      : gamma_(gamma), threshold_(threshold) {}
  DPAgent(const DPAgent &) = default;
  DPAgent(DPAgent &&) = default;
  DPAgent &operator=(const DPAgent &) = default;
  DPAgent &operator=(DPAgent &&) = default;
  ~DPAgent() override = default;
  std::shared_ptr<Agent> Clone() const & override {
    return std::shared_ptr<Agent>(new DPAgent(*this));
  }
  std::shared_ptr<Agent> Clone() && override {
    return std::shared_ptr<Agent>(new DPAgent(std::move(*this)));
  }
  double GetGamma() const { return gamma_; }
  double GetThreshold() const { return threshold_; }
  std::unordered_map<StateAction, std::vector<StateProb>>
  GetTransitions() const { return transitions_; }
  void Initialize(const std::vector<State> &) override;
  void SetGamma(double gamma) { gamma_ = gamma; }
  void SetThreshold(double threshold) { threshold_ = threshold; }
  template<typename T>
  void SetTransitions(T &&transitions) {
    transitions_ = std::forward<T>(transitions);
  }

 protected:
  std::unordered_map<StateAction, std::vector<StateProb>> transitions_;
  double gamma_;
  double threshold_;

 private:
  Action PolicyImpl(const std::vector<Action> &/*legal_actions*/,
                    const std::vector<Action> &greedy_actions) override {
    return SampleAction(greedy_actions);
  }
};

class PolicyIterationAgent : public DPAgent {
 public:
  explicit PolicyIterationAgent(double gamma = kDefaultGamma,
                                double threshold = kDefaultThreshold)
      : DPAgent(gamma, threshold) {}
  PolicyIterationAgent(const PolicyIterationAgent &) = default;
  PolicyIterationAgent(PolicyIterationAgent &&) = default;
  PolicyIterationAgent &operator=(const PolicyIterationAgent &) = default;
  PolicyIterationAgent &operator=(PolicyIterationAgent &&) = default;
  ~PolicyIterationAgent() override = default;
  std::shared_ptr<Agent> Clone() const & override {
    return std::shared_ptr<Agent>(new PolicyIterationAgent(*this));
  }
  std::shared_ptr<Agent> Clone() && override {
    return std::shared_ptr<Agent>(new PolicyIterationAgent(std::move(*this)));
  }
  void Initialize(const std::vector<State> &) override;

 private:
  std::unordered_map<State, Action> policy_;
  Action Policy(const State &state, bool is_evaluation) override {
    return is_evaluation ? DPAgent::Policy(state, is_evaluation)
                         : policy_[state];
  }
  void PolicyIteration(const std::vector<State> &);
};

class ValueIterationAgent : public DPAgent {
 public:
  explicit ValueIterationAgent(double gamma = kDefaultGamma,
                               double threshold = kDefaultThreshold)
      : DPAgent(gamma, threshold) {}
  ValueIterationAgent(const ValueIterationAgent &) = default;
  ValueIterationAgent(ValueIterationAgent &&) = default;
  ValueIterationAgent &operator=(const ValueIterationAgent &) = default;
  ValueIterationAgent &operator=(ValueIterationAgent &&) = default;
  ~ValueIterationAgent() override = default;
  std::shared_ptr<Agent> Clone() const & override {
    return std::shared_ptr<Agent>(new ValueIterationAgent(*this));
  }
  std::shared_ptr<Agent> Clone() && override {
    return std::shared_ptr<Agent>(new ValueIterationAgent(std::move(*this)));
  }
  void Initialize(const std::vector<State> &) override;

 private:
  void ValueIteration(const std::vector<State> &);
};

}  // namespace nim_rl

#endif  // NIM_RL_AGENT_DP_AGENT_H_
