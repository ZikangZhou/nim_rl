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

#ifndef NIM_RL_ACTION_ACTION_H_
#define NIM_RL_ACTION_ACTION_H_

#include <iostream>
#include <sstream>
#include <string>

namespace nim_rl {

class State;

class Action {
  friend class std::hash<Action>;

 public:
  Action() = default;
  Action(int pile_id, int num_objects)
      : pile_id_(pile_id), num_objects_(num_objects) {}
  explicit Action(std::istream &);
  Action(const Action &) = default;
  Action(Action &&action) noexcept
      : pile_id_(action.pile_id_), num_objects_(action.num_objects_) {}
  Action &operator=(const Action &) = default;
  Action &operator=(Action &&) noexcept;
  ~Action() = default;
  int GetNumObjects() const { return num_objects_; }
  int GetPileId() const { return pile_id_; }
  bool IsLegal(const State &) const;
  void SetNumObjects(int num_object) { num_objects_ = num_object; }
  void SetPileId(int pile_id) { pile_id_ = pile_id; }
  std::string ToString() const {
    return "<" + std::to_string(pile_id_) + ", " + std::to_string(num_objects_)
        + ">";
  }

 private:
  int pile_id_ = -1;
  int num_objects_ = -1;
};

std::istream &operator>>(std::istream &, Action &);
std::ostream &operator<<(std::ostream &, const Action &);
bool operator==(const Action &, const Action &);
bool operator!=(const Action &, const Action &);

}  // namespace nim_rl

namespace std {
using nim_rl::Action;
template<>
struct hash<Action> {
  std::size_t operator()(const Action &action) const {
    std::size_t seed = 0;
    seed ^= std::hash<int>()(action.pile_id_) + 0x9e3779b9
        + (seed << 6u) + (seed >> 2u);
    seed ^= std::hash<int>()(action.num_objects_) + 0x9e3779b9
        + (seed << 6u) + (seed >> 2u);
    return seed;
  }
};
}  // namespace std

#endif  // NIM_RL_ACTION_ACTION_H_
