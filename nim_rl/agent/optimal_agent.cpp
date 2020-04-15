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

#include "nim_rl/agent/optimal_agent.h"

namespace nim_rl {

Action OptimalAgent::Policy(const State &state, bool /*is_evaluation*/) {
  unsigned nim_sum = state.NimSum();
  for (int pile_id = 0; pile_id != state.Size(); ++pile_id) {
    unsigned num_objects_target = state[pile_id] ^nim_sum;
    if (num_objects_target < state[pile_id])
      return Action{pile_id,
                    static_cast<int>(state[pile_id] - num_objects_target)};
  }
  return SampleAction(state.LegalActions());
}

}  // namespace nim_rl
