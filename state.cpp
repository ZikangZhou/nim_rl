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
    if (!state_stream) {
      is.setstate(is.rdstate() | std::istream::failbit);
      state_.clear();
      std::cerr << "Input Error: Input must be integer." << std::endl;
    }
  }
  if (!is) {
    std::cerr << "Warning: Input stream has failed." << std::endl;
  }
  CheckEmpty("Warning: Input state is empty.");
}

State::State(const State &state) : state_(state.state_) {
  CheckEmpty("Warning: State is empty.");
}

State::State(std::initializer_list<unsigned> il) : state_(il) {
  CheckEmpty("Warning: State is empty.");
}

State::State(const std::vector<unsigned> &vec) : state_(vec) {
  CheckEmpty("Warning: State is empty.");
}

State::State(State &&state) noexcept : state_(std::move(state.state_)) {
  CheckEmpty("Warning: State is empty.");
  state.state_ = {};
}

State &State::operator=(const State &rhs) {
  state_ = rhs.state_;
  CheckEmpty("Warning: State is empty.");
  return *this;
}

State &State::operator=(State &&rhs) noexcept {
  state_ = std::move(rhs.state_);
  CheckEmpty("Warning: State is empty.");
  return *this;
}

State &State::operator=(std::initializer_list<unsigned> il) {
  state_ = il;
  CheckEmpty("Warning: State is empty.");
  return *this;
}

State &State::operator=(const std::vector<unsigned> &vec) {
  state_ = vec;
  CheckEmpty("Warning: State is empty.");
  return *this;
}

void State::RemoveObjects(size_type pile_id, unsigned num_objects) {
  CheckRange(pile_id, "Pile_id is out of range.");
  if (num_objects > state_[pile_id] || num_objects < 1) {
    throw std::out_of_range("num_objects must be [1, State[pile_id]]");
  }
  state_[pile_id] -= num_objects;
}

unsigned &State::operator[](size_type pile_id) {
  CheckRange(pile_id, "Pile_id is out of range.");
  return state_[pile_id];
}

const unsigned &State::operator[](size_type pile_id) const {
  CheckRange(pile_id, "Pile_id is out of range.");
  return state_[pile_id];
}

std::istream &operator>>(std::istream &is, State &state) {
  State new_state(is);
  if (is) {
    state = new_state;
  } else {
    is.clear();
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

bool operator==(const State &lhs, const State &rhs) {
  return lhs.state_ == rhs.state_;
}

bool operator!=(const State &lhs, const State &rhs) {
  return !(lhs == rhs);
}
