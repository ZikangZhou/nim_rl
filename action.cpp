//
// Created by 周梓康 on 2020/3/8.
//

#include "action.h"
#include "state.h"

Action &Action::operator=(Action &&rhs) noexcept {
  pile_id_ = rhs.pile_id_;
  num_objects_ = rhs.num_objects_;
  return *this;
}

bool Action::Valid(const State &state) const {
  return pile_id_ >= 0 && pile_id_ <= state.Size() - 1 && num_objects_ >= 1
      && num_objects_ <= state[pile_id_];
}
std::istream &operator>>(std::istream &is, Action &action) {
  Action tmp;
  std::string line;
  if (getline(is, line)) {
    std::istringstream state_stream(line);
    state_stream >> tmp.pile_id_ >> tmp.num_objects_;
    if (!state_stream.eof()) {
      is.setstate(is.rdstate() | std::istream::failbit);
    } else {
      action = tmp;
    }
  }
  if (!is) {
    std::cerr << "Error: Invalid input." << std::endl;
  }
  return is;
}

std::ostream &operator<<(std::ostream &os, const Action &action) {
  os << "Action(" << action.pile_id_ << ", " << action.num_objects_ << ")";
  return os;
}
