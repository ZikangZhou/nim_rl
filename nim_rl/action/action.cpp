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

#include "nim_rl/action/action.h"
#include "nim_rl/state/state.h"

namespace nim_rl {

Action::Action(std::istream &is) {
  std::string line;
  if (getline(is, line)) {
    std::istringstream iss(line);
    iss >> pile_id_ >> num_objects_;
    if (!iss.eof()) {
      is.setstate(is.rdstate() | std::istream::failbit);
      pile_id_ = num_objects_ = -1;
    }
  }
  if (!is) std::cerr << "Error: Invalid input." << std::endl;
}

Action &Action::operator=(Action &&rhs) noexcept {
  pile_id_ = rhs.pile_id_;
  num_objects_ = rhs.num_objects_;
  return *this;
}

bool Action::IsLegal(const State &state) const {
  return pile_id_ >= 0 && pile_id_ <= state.Size() - 1 && num_objects_ >= 1 &&
      num_objects_ <= state[pile_id_];
}

std::istream &operator>>(std::istream &is, Action &action) {
  Action action_is(is);
  if (is) action = std::move(action_is);
  return is;
}

std::ostream &operator<<(std::ostream &os, const Action &action) {
  os << action.ToString();
  return os;
}

bool operator==(const Action &lhs, const Action &rhs) {
  return lhs.GetPileId() == rhs.GetPileId() &&
      lhs.GetNumObjects() == rhs.GetNumObjects();
}

bool operator!=(const Action &lhs, const Action &rhs) {
  return !(lhs == rhs);
}

}  // namespace nim_rl
