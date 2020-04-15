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

#ifndef NIM_RL_AGENT_N_STEP_BOOTSTRAPPING_AGENT_H_
#define NIM_RL_AGENT_N_STEP_BOOTSTRAPPING_AGENT_H_

#include "nim_rl/agent/td_agent.h"

namespace nim_rl {

constexpr int kDefaultN = 1;

class NStepBootstrappingAgent : public TDAgent {
 public:
  explicit NStepBootstrappingAgent(
      double alpha = kDefaultAlpha,
      double gamma = kDefaultGamma,
      int n = kDefaultN,
      double epsilon = kDefaultEpsilon,
      double epsilon_decay_factor = kDefaultEpsilonDecayFactor,
      double min_epsilon = kDefaultMinEpsilon)
      : TDAgent(alpha, gamma, epsilon, epsilon_decay_factor, min_epsilon),
        n_(n) {}
  NStepBootstrappingAgent(const NStepBootstrappingAgent &) = default;
  NStepBootstrappingAgent(NStepBootstrappingAgent &&) = default;
  NStepBootstrappingAgent &operator=(const NStepBootstrappingAgent &) = default;
  NStepBootstrappingAgent &operator=(NStepBootstrappingAgent &&) = default;
  ~NStepBootstrappingAgent() override = default;
  int GetN() const { return n_; }
  void Reset() override;
  void SetN(int n) { n_ = n; }
  Action Step(Game *, bool is_evaluation) override;

 protected:
  int n_;
  int current_time_ = 0;
  int terminal_time_ = INT_MAX;
  int update_time_ = 0;
  std::vector<TimeStep> trajectory_;
};

class NStepSarsaAgent : public NStepBootstrappingAgent {
 public:
  explicit NStepSarsaAgent(
      double alpha = kDefaultAlpha,
      double gamma = kDefaultGamma,
      int n = kDefaultN,
      double epsilon = kDefaultEpsilon,
      double epsilon_decay_factor = kDefaultEpsilonDecayFactor,
      double min_epsilon = kDefaultMinEpsilon)
      : NStepBootstrappingAgent(alpha, gamma, n, epsilon, epsilon_decay_factor,
                                min_epsilon) {}
  NStepSarsaAgent(const NStepSarsaAgent &) = default;
  NStepSarsaAgent(NStepSarsaAgent &&) = default;
  NStepSarsaAgent &operator=(const NStepSarsaAgent &) = default;
  NStepSarsaAgent &operator=(NStepSarsaAgent &&) = default;
  ~NStepSarsaAgent() override = default;
  std::shared_ptr<Agent> Clone() const & override {
    return std::shared_ptr<Agent>(new NStepSarsaAgent(*this));
  }
  std::shared_ptr<Agent> Clone() && override {
    return std::shared_ptr<Agent>(new NStepSarsaAgent(std::move(*this)));
  }

 protected:
  void Update(const State &update_state, const State &current_state,
              Reward reward) override;
};

class NStepExpectedSarsaAgent : public NStepBootstrappingAgent {
 public:
  explicit NStepExpectedSarsaAgent(
      double alpha = kDefaultAlpha,
      double gamma = kDefaultGamma,
      int n = kDefaultN,
      double epsilon = kDefaultEpsilon,
      double epsilon_decay_factor = kDefaultEpsilonDecayFactor,
      double min_epsilon = kDefaultMinEpsilon)
      : NStepBootstrappingAgent(alpha, gamma, n, epsilon, epsilon_decay_factor,
                                min_epsilon) {}
  NStepExpectedSarsaAgent(const NStepExpectedSarsaAgent &) = default;
  NStepExpectedSarsaAgent(NStepExpectedSarsaAgent &&) = default;
  NStepExpectedSarsaAgent &operator=(const NStepExpectedSarsaAgent &) = default;
  NStepExpectedSarsaAgent &operator=(NStepExpectedSarsaAgent &&) = default;
  ~NStepExpectedSarsaAgent() override = default;
  std::shared_ptr<Agent> Clone() const & override {
    return std::shared_ptr<Agent>(new NStepExpectedSarsaAgent(*this));
  }
  std::shared_ptr<Agent> Clone() && override {
    return std::shared_ptr<Agent>(
        new NStepExpectedSarsaAgent(std::move(*this)));
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

class OffPolicyNStepSarsaAgent : public NStepBootstrappingAgent {
 public:
  explicit OffPolicyNStepSarsaAgent(
      double alpha = kDefaultAlpha,
      double gamma = kDefaultGamma,
      int n = kDefaultN,
      double epsilon = kDefaultEpsilon,
      double epsilon_decay_factor = kDefaultEpsilonDecayFactor,
      double min_epsilon = kDefaultMinEpsilon)
      : NStepBootstrappingAgent(alpha, gamma, n, epsilon, epsilon_decay_factor,
                                min_epsilon) {}
  OffPolicyNStepSarsaAgent(const OffPolicyNStepSarsaAgent &) = default;
  OffPolicyNStepSarsaAgent(OffPolicyNStepSarsaAgent &&) = default;
  OffPolicyNStepSarsaAgent &
  operator=(const OffPolicyNStepSarsaAgent &) = default;
  OffPolicyNStepSarsaAgent &operator=(OffPolicyNStepSarsaAgent &&) = default;
  ~OffPolicyNStepSarsaAgent() override = default;
  std::shared_ptr<Agent> Clone() const & override {
    return std::shared_ptr<Agent>(new OffPolicyNStepSarsaAgent(*this));
  }
  std::shared_ptr<Agent> Clone() && override {
    return std::shared_ptr<Agent>(
        new OffPolicyNStepSarsaAgent(std::move(*this)));
  }

 protected:
  void Update(const State &update_state, const State &current_state,
              Reward reward) override;
};

class OffPolicyNStepExpectedSarsaAgent : public NStepBootstrappingAgent {
 public:
  explicit OffPolicyNStepExpectedSarsaAgent(
      double alpha = kDefaultAlpha,
      double gamma = kDefaultGamma,
      int n = kDefaultN,
      double epsilon = kDefaultEpsilon,
      double epsilon_decay_factor = kDefaultEpsilonDecayFactor,
      double min_epsilon = kDefaultMinEpsilon)
      : NStepBootstrappingAgent(alpha, gamma, n, epsilon, epsilon_decay_factor,
                                min_epsilon) {}
  OffPolicyNStepExpectedSarsaAgent(
      const OffPolicyNStepExpectedSarsaAgent &) = default;
  OffPolicyNStepExpectedSarsaAgent(
      OffPolicyNStepExpectedSarsaAgent &&) = default;
  OffPolicyNStepExpectedSarsaAgent &
  operator=(const OffPolicyNStepExpectedSarsaAgent &) = default;
  OffPolicyNStepExpectedSarsaAgent &
  operator=(OffPolicyNStepExpectedSarsaAgent &&) = default;
  ~OffPolicyNStepExpectedSarsaAgent() override = default;
  std::shared_ptr<Agent> Clone() const & override {
    return std::shared_ptr<Agent>(new OffPolicyNStepExpectedSarsaAgent(*this));
  }
  std::shared_ptr<Agent> Clone() && override {
    return std::shared_ptr<Agent>(
        new OffPolicyNStepExpectedSarsaAgent(std::move(*this)));
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

class NStepTreeBackupAgent : public NStepBootstrappingAgent {
 public:
  explicit NStepTreeBackupAgent(
      double alpha = kDefaultAlpha,
      double gamma = kDefaultGamma,
      int n = kDefaultN,
      double epsilon = kDefaultEpsilon,
      double epsilon_decay_factor = kDefaultEpsilonDecayFactor,
      double min_epsilon = kDefaultMinEpsilon)
      : NStepBootstrappingAgent(alpha, gamma, n, epsilon, epsilon_decay_factor,
                                min_epsilon) {}
  NStepTreeBackupAgent(const NStepTreeBackupAgent &) = default;
  NStepTreeBackupAgent(NStepTreeBackupAgent &&) = default;
  NStepTreeBackupAgent &operator=(const NStepTreeBackupAgent &) = default;
  NStepTreeBackupAgent &operator=(NStepTreeBackupAgent &&) = default;
  ~NStepTreeBackupAgent() override = default;
  std::shared_ptr<Agent> Clone() const & override {
    return std::shared_ptr<Agent>(new NStepTreeBackupAgent(*this));
  }
  std::shared_ptr<Agent> Clone() && override {
    return std::shared_ptr<Agent>(new NStepTreeBackupAgent(std::move(*this)));
  }

 protected:
  void Update(const State &update_state, const State &current_state,
              Reward reward) override;
};

}  // namespace nim_rl

#endif  // NIM_RL_AGENT_N_STEP_BOOTSTRAPPING_AGENT_H_
