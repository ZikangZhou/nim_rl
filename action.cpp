//
// Created by 周梓康 on 2020/3/8.
//

#include "action.h"

Action &Action::operator=(Action &&rhs) noexcept {
  action_ = std::move(rhs.action_);
  return *this;
}

std::istream &operator>>(std::istream &is, Action &action) {
  Action tmp;
  std::string line;
  if (getline(is, line)) {
    std::istringstream state_stream(line);
    state_stream >> tmp.action_.first >> tmp.action_.second;
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
  os << "Action(" << action.action_.first << ", " << action.action_.second
     << ")";
  return os;
}
