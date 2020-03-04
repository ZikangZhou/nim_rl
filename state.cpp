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
  }
  if (state_.empty()) {
    throw std::invalid_argument("State should not be empty");
  }
}

State::State(std::initializer_list<unsigned> state) {
  for (auto num_objects : state) {
    state_.push_back(num_objects);
  }
  if (state_.empty()) {
    throw std::invalid_argument("State should not be empty");
  }
}

State::State(std::vector<unsigned> state) : state_(std::move(state)) {
  if (state_.empty()) {
    throw std::invalid_argument("State should not be empty");
  }
}

State &State::operator=(std::initializer_list<unsigned> state) {
  state_.clear();
  for (auto num_objects : state) {
    state_.push_back(num_objects);
  }
  if (state_.empty()) {
    throw std::invalid_argument("State should not be empty");
  }
  return *this;
}

State &State::operator=(const std::vector<unsigned> &state) {
  state_ = state;
  if (state_.empty()) {
    throw std::invalid_argument("State should not be empty");
  }
  return *this;
}

void State::RemoveObjects(std::vector<unsigned>::size_type pile_id,
                          unsigned num_objects) {
  if (pile_id >= state_.size()) {
    throw std::out_of_range("pile_id should not be out of range");
  }
  if (num_objects > state_[pile_id] || num_objects < 1) {
    throw std::out_of_range("num_objects must be [1, State[pile_id]]");
  }
  state_[pile_id] -= num_objects;
}

unsigned &State::operator[](std::vector<unsigned>::size_type pile_id) {
  if (pile_id >= state_.size()) {
    throw std::out_of_range("pile_id should not be out of range");
  }
  return state_[pile_id];
}

const unsigned &State::operator[](
    std::vector<unsigned>::size_type pile_id) const {
  if (pile_id >= state_.size()) {
    throw std::out_of_range("pile_id should not be out of range");
  }
  return state_[pile_id];
}

std::istream &operator>>(std::istream &is, State &state) {
  state.clear();
  std::string line;
  unsigned num_objects;
  if (getline(is, line)) {
    std::istringstream state_stream(line);
    while (state_stream >> num_objects) {
      state.push_back(num_objects);
    }
  }
  if (state.empty()) {
    throw std::invalid_argument("State should not be empty");
  }
  return is;
}

std::ostream &operator<<(std::ostream &os, const State &state) {
  if (!state.empty()) {
    for (decltype(state.size()) pile_id = 0; pile_id != state.size() - 1;
         ++pile_id) {
      os << state[pile_id] << " ";
    }
    os << state[state.size() - 1];
  }
  return os;
}
