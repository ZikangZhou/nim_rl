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

#ifndef NIM_RL_AGENT_RL_AGENT_H_
#define NIM_RL_AGENT_RL_AGENT_H_

#include "nim_rl/agent/agent.h"

namespace nim_rl {

constexpr double kDefaultAlpha = 0.5;
constexpr double kDefaultGamma = 1.0;

class RLAgent : public Agent {
 public:
  using StateAction = std::pair<State, Action>;
  using StateProb = std::pair<State, double>;
  using TimeStep = std::tuple<State, Action, Reward>;
  using Values = std::unordered_map<State, Reward>;
  RLAgent() = default;
  RLAgent(const RLAgent &) = default;
  RLAgent(RLAgent &&) = default;
  RLAgent &operator=(const RLAgent &) = default;
  RLAgent &operator=(RLAgent &&) = default;
  ~RLAgent() override = default;
  void AddGreedyAction(const Action &action) {
    greedy_actions_.push_back(action);
  }
  void ClearGreedyActions() { greedy_actions_.clear(); }
  std::vector<Action> GetGreedyActions() { return greedy_actions_; }
  Reward GetGreedyValue() const { return greedy_value_; }
  std::vector<Action> GetLegalActions() const { return legal_actions_; }
  virtual Values GetValues() const { return *values_; }
  void Initialize(const std::vector<State> &) override;
  double MinSquareError();
  double OptimalActionsRatio();
  Action Policy(const State &, bool is_evaluation) override;
  virtual Action PolicyImpl(const std::vector<Action> &legal_actions,
                            const std::vector<Action> &greedy_actions) = 0;
  void Reset() override;
  void SetGreedyActions(const std::vector<Action> &greedy_actions) {
    greedy_actions_ = greedy_actions;
  }
  void SetGreedyValue(Reward greedy_value) { greedy_value_ = greedy_value; }
  void SetLegalActions(const std::vector<Action> &legal_actions) {
    legal_actions_ = legal_actions;
  }
  virtual void SetValues(const Values &values) { *values_ = values; }
  virtual void UpdateExploration(int episode) {}

 protected:
  std::shared_ptr<Values> values_ = std::shared_ptr<Values>(new Values());
  Reward greedy_value_ = 0.0;
  std::vector<Action> legal_actions_;
  std::vector<Action> greedy_actions_;
};

std::ostream &operator<<(std::ostream &,
                         const std::unordered_map<State, Agent::Reward> &);

std::ostream &operator<<(std::ostream &,
                         const std::vector<RLAgent::TimeStep> &);

}  // namespace nim_rl

#endif  // NIM_RL_AGENT_RL_AGENT_H_
