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

#ifndef NIM_RL_AGENT_HUMAN_AGENT_H_
#define NIM_RL_AGENT_HUMAN_AGENT_H_

#include "nim_rl/agent/agent.h"

namespace nim_rl {

class HumanAgent : public Agent {
 public:
  HumanAgent() = default;
  HumanAgent(const HumanAgent &) = default;
  HumanAgent(HumanAgent &&) = default;
  HumanAgent &operator=(const HumanAgent &) = default;
  HumanAgent &operator=(HumanAgent &&) = default;
  ~HumanAgent() override = default;
  std::shared_ptr<Agent> Clone() const override {
    return std::shared_ptr<Agent>(new HumanAgent(*this));
  }
  Action Policy(const State &, bool is_evaluation) override;
};

}  // namespace nim_rl

#endif  // NIM_RL_AGENT_HUMAN_AGENT_H_
