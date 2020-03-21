//
// Created by 周梓康 on 2020/3/3.
//

#include "state.h"

State::State(std::istream &is) {
  std::string line;
  unsigned num_objects;
  if (getline(is, line)) {
    std::istringstream state_stream(line);
    while (state_stream >> num_objects) {
      data_.push_back(num_objects);
    }
    if (!state_stream.eof()) {
      is.setstate(is.rdstate() | std::istream::failbit);
      data_.clear();
    }
  }
  if (!is) {
    std::cerr << "Error: Invalid Input." << std::endl;
  }
  CheckEmpty();
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

std::vector<Action> State::ActionSpace() const {
  std::vector<Action> action_space;
  for (int pile_id = 0; pile_id != data_.size(); ++pile_id) {
    for (int num_objects = 1; num_objects != data_[pile_id] + 1;
         ++num_objects) {
      action_space.emplace_back(pile_id, num_objects);
    }
  }
  return action_space;
}

bool State::End() const {
  for (int pile_id = 0; pile_id != data_.size(); ++pile_id) {
    if (data_[pile_id]) {
      return false;
    }
  }
  return true;
}

State State::Next(const Action &action) const {
  int pile_id = action.pile_id();
  int num_objects = action.num_objects();
  CheckRange(pile_id);
  if (num_objects > data_[pile_id] || num_objects < 1) {
    throw std::out_of_range("num_objects must fall in [1, State[pile_id]]");
  }
  State next_state(*this);
  next_state.data_[pile_id] -= num_objects;
  return next_state;
}

unsigned State::NimSum() const {
  unsigned nim_sum = 0;
  for (int pile_id = 0; pile_id != data_.size(); ++pile_id) {
    nim_sum ^= data_[pile_id];
  }
  return nim_sum;
}

void State::Update(const Action &action) {
  int pile_id = action.pile_id();
  int num_objects = action.num_objects();
  CheckRange(pile_id);
  if (num_objects > data_[pile_id] || num_objects < 1) {
    throw std::out_of_range("num_objects must fall in [1, State[pile_id]]");
  }
  data_[pile_id] -= num_objects;
}

unsigned &State::operator[](int pile_id) {
  CheckRange(pile_id);
  return data_[pile_id];
}

const unsigned &State::operator[](int pile_id) const {
  CheckRange(pile_id);
  return data_[pile_id];
}

std::istream &operator>>(std::istream &is, State &state) {
  State tmp(is);
  if (is) {
    state = std::move(tmp);
  }
  return is;
}

std::ostream &operator<<(std::ostream &os, const State &state) {
  os << "{";
  if (!state.Empty()) {
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
