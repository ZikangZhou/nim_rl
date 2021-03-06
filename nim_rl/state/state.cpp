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

#include "nim_rl/state/state.h"

namespace nim_rl {

State::State(std::istream &is) {
  std::string line;
  unsigned num_objects;
  if (getline(is, line)) {
    std::istringstream iss(line);
    while (iss >> num_objects) data_.push_back(num_objects);
    if (!iss.eof()) {
      is.setstate(is.rdstate() | std::istream::failbit);
      data_.clear();
    }
  }
  if (!is) std::cerr << "Error: Invalid Input." << std::endl;
}

State &State::operator=(State &&rhs) noexcept {
  data_ = std::move(rhs.data_);
  return *this;
}

State &State::operator=(std::initializer_list<unsigned> il) {
  data_ = il;
  return *this;
}

State &State::operator=(std::vector<unsigned> vec) {
  data_ = std::move(vec);
  return *this;
}

void State::ApplyAction(const Action &action) {
  int pile_id = action.GetPileId();
  CheckRange(pile_id);
  int num_objects = action.GetNumObjects();
  if (num_objects > data_[pile_id] || num_objects < 1)
    throw std::out_of_range("num_objects must fall in [1, State[pile_id]]");
  data_[pile_id] -= num_objects;
}

State State::Child(const Action &action) const {
  if (IsTerminal()) {
    return State();
  } else {
    if (!action.IsLegal(*this)) {
      return *this;
    } else {
      State child(*this);
      child.ApplyAction(action);
      return child;
    }
  }
}

std::vector<State> State::Children() const {
  std::vector<State> children;
  State state(*this);
  if (state.IsTerminal()) {
    children.emplace_back();
  } else {
    for (int pile_id = 0; pile_id != data_.size(); ++pile_id) {
      for (int num_objects = 0; num_objects != data_[pile_id]; ++num_objects) {
        state[pile_id] = static_cast<unsigned>(num_objects);
        children.push_back(state);
      }
      state[pile_id] = data_[pile_id];
    }
  }
  return children;
}

std::vector<State> State::GetAllStates() const {
  std::vector<unsigned> initial_state(data_);
  std::sort(initial_state.begin(), initial_state.end());
  std::vector<State> all_states;
  all_states.reserve(initial_state[0] + 1);
  for (int num_objects = 0; num_objects != initial_state[0] + 1;
       ++num_objects)
    all_states.emplace_back(1, num_objects);
  std::vector<State> new_all_states;
  for (int pile_id = 1; pile_id < initial_state.size(); ++pile_id) {
    for (const auto &state : all_states) {
      for (unsigned num_objects = state.data_.back();
           num_objects != initial_state[pile_id] + 1; ++num_objects) {
        State next_state(state);
        next_state.data_.push_back(num_objects);
        new_all_states.emplace_back(std::move(next_state));
      }
    }
    std::swap(all_states, new_all_states);
    new_all_states.clear();
  }
  return all_states;
}

bool State::IsTerminal() const {
  for (int pile_id = 0; pile_id != data_.size(); ++pile_id)
    if (data_[pile_id]) return false;
  return true;
}

std::vector<Action> State::LegalActions() const {
  std::vector<Action> legal_actions;
  for (int pile_id = 0; pile_id != data_.size(); ++pile_id)
    for (int num_objects = 1; num_objects != data_[pile_id] + 1;
         ++num_objects)
      legal_actions.emplace_back(pile_id, num_objects);
  return legal_actions;
}

unsigned State::NimSum() const {
  unsigned nim_sum = 0;
  for (int pile_id = 0; pile_id != data_.size(); ++pile_id)
    nim_sum ^= data_[pile_id];
  return nim_sum;
}

State State::Parent(const Action &action) const {
  State parent(*this);
  parent.UndoAction(action);
  return parent;
}

std::string State::ToString() const {
  std::ostringstream oss;
  oss << "[";
  if (!data_.empty()) {
    for (int pile_id = 0; pile_id != data_.size() - 1; ++pile_id)
      oss << data_[pile_id] << ", ";
    oss << data_[static_cast<int>(data_.size()) - 1];
  }
  oss << "]";
  return oss.str();
}

void State::UndoAction(const Action &action) {
  int pile_id = action.GetPileId();
  CheckRange(pile_id);
  int num_objects = action.GetNumObjects();
  if (num_objects < 1)
    throw std::out_of_range("num_objects must be no less than 1");
  data_[pile_id] += num_objects;
}

unsigned &State::operator[](int pile_id) {
  CheckRange(pile_id);
  return data_[pile_id];
}

const unsigned &State::operator[](int pile_id) const {
  CheckRange(pile_id);
  return data_[pile_id];
}

void State::DoGetAllStates(const State &state, int pile_id,
                           std::vector<State> *all_states) const {
  if (pile_id == data_.size()) {
    if (std::find(all_states->begin(), all_states->end(), state) ==
        all_states->end())
      all_states->push_back(state);
    return;
  }
  Action action(pile_id, -1);
  DoGetAllStates(state, pile_id + 1, all_states);
  for (int num_objects = 1; num_objects != state[pile_id] + 1; ++num_objects) {
    action.SetNumObjects(num_objects);
    DoGetAllStates(state.Child(action), pile_id + 1, all_states);
  }
}

std::istream &operator>>(std::istream &is, State &state) {
  State state_is(is);
  if (is) state = std::move(state_is);
  return is;
}

std::ostream &operator<<(std::ostream &os, const State &state) {
  os << state.ToString();
  return os;
}

bool operator==(const State &lhs, const State &rhs) {
  std::vector<unsigned> lhs_sorted(lhs.data_), rhs_sorted(rhs.data_);
  std::sort(lhs_sorted.begin(), lhs_sorted.end());
  std::sort(rhs_sorted.begin(), rhs_sorted.end());
  return lhs_sorted == rhs_sorted;
}

bool operator!=(const State &lhs, const State &rhs) {
  return !(lhs == rhs);
}

}  // namespace nim_rl
