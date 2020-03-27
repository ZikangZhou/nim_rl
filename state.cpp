//
// Created by 周梓康 on 2020/3/3.
//

#include "state.h"

State::State(std::istream &is) {
  std::string line;
  unsigned num_objects;
  if (getline(is, line)) {
    std::istringstream iss(line);
    while (iss >> num_objects) {
      data_.push_back(num_objects);
    }
    if (!iss.eof()) {
      is.setstate(is.rdstate() | std::istream::failbit);
      data_.clear();
    }
  }
  if (!is) {
    std::cerr << "Error: Invalid Input." << std::endl;
  }
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
  if (num_objects > data_[pile_id] || num_objects < 1) {
    throw std::out_of_range("num_objects must fall in [1, State[pile_id]]");
  }
  data_[pile_id] -= num_objects;
}

State State::Child(const Action &action) const {
  State child(*this);
  child.ApplyAction(action);
  return child;
}

std::vector<State> State::Children() const {
  std::vector<State> children;
  State state(*this);
  for (int pile_id = 0; pile_id != data_.size(); ++pile_id) {
    for (int num_objects = 0; num_objects != data_[pile_id]; ++num_objects) {
      state[pile_id] = static_cast<unsigned>(num_objects);
      children.push_back(state);
    }
    state[pile_id] = data_[pile_id];
  }
  return children;
}

std::vector<State> State::GetAllStates() const {
  std::vector<State> all_states;
  DoGetAllStates(*this, 0, &all_states);
  return all_states;
}

bool State::IsTerminal() const {
  for (int pile_id = 0; pile_id != data_.size(); ++pile_id) {
    if (data_[pile_id]) {
      return false;
    }
  }
  return true;
}

std::vector<Action> State::LegalActions() const {
  std::vector<Action> legal_actions;
  for (int pile_id = 0; pile_id != data_.size(); ++pile_id) {
    for (int num_objects = 1; num_objects != data_[pile_id] + 1;
         ++num_objects) {
      legal_actions.emplace_back(pile_id, num_objects);
    }
  }
  return legal_actions;
}

unsigned State::NimSum() const {
  unsigned nim_sum = 0;
  for (int pile_id = 0; pile_id != data_.size(); ++pile_id) {
    nim_sum ^= data_[pile_id];
  }
  return nim_sum;
}

State State::Parent(const Action &action) const {
  State parent(*this);
  parent.UndoAction(action);
  return parent;
}

void State::UndoAction(const Action &action) {
  int pile_id = action.GetPileId();
  CheckRange(pile_id);
  int num_objects = action.GetNumObjects();
  if (num_objects < 1) {
    throw std::out_of_range("num_objects must be no less than 1");
  }
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

void State::DoGetAllStates(const State &state,
                           int pile_id,
                           std::vector<State> *all_states) const {
  if (pile_id == data_.size()) {
    if (std::find(all_states->begin(), all_states->end(), state)
        == all_states->end()) {
      all_states->push_back(state);
    }
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
  if (is) {
    state = std::move(state_is);
  }
  return is;
}

std::ostream &operator<<(std::ostream &os, const State &state) {
  os << "{";
  if (!state.IsEmpty()) {
    for (int pile_id = 0; pile_id != state.Size() - 1; ++pile_id) {
      os << state[pile_id] << ", ";
    }
    os << state[state.Size() - 1];
  }
  os << "}";
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
