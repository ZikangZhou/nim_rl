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
      state_.push_back(num_objects);
    }
    if (!state_stream.eof()) {
      is.setstate(is.rdstate() | std::istream::failbit);
      state_.clear();
    }
  }
  if (!is) {
    std::cerr << "Error: Invalid Input." << std::endl;
  }
  CheckEmpty();
}

State &State::operator=(const State &rhs) {
  state_ = rhs.state_;
  CheckEmpty();
  return *this;
}

State &State::operator=(State &&rhs) noexcept {
  state_ = std::move(rhs.state_);
  CheckEmpty();
  return *this;
}

State &State::operator=(std::initializer_list<unsigned> il) {
  state_ = il;
  CheckEmpty();
  return *this;
}

State &State::operator=(std::vector<unsigned> vec) {
  state_ = std::move(vec);
  CheckEmpty();
  return *this;
}

void State::TakeAction(const Action &action) {
  int pile_id = action.pile_id();
  int num_objects = action.num_objects();
  CheckRange(pile_id);
  if (num_objects > state_[pile_id] || num_objects < 1) {
    throw std::out_of_range("num_objects must be [1, State[pile_id]]");
  }
  state_[pile_id] -= num_objects;
}

unsigned &State::operator[](int pile_id) {
  CheckRange(pile_id);
  return state_[pile_id];
}

const unsigned &State::operator[](int pile_id) const {
  CheckRange(pile_id);
  return state_[pile_id];
}

std::istream &operator>>(std::istream &is, State &state) {
  State tmp(is);
  if (is) {
    state = std::move(tmp);
  } else {
    is.clear();
  }
  return is;
}

std::ostream &operator<<(std::ostream &os, const State &state) {
  os << "State{";
  if (!state.Empty()) {
    for (decltype(state.Size()) pile_id = 0; pile_id != state.Size() - 1;
         ++pile_id) {
      os << state[pile_id] << ", ";
    }
    os << state[state.Size() - 1];
  }
  os << "}";
  return os;
}

bool operator==(const State &lhs, const State &rhs) {
  return lhs.state_ == rhs.state_;
}

bool operator!=(const State &lhs, const State &rhs) {
  return !(lhs == rhs);
}
