//
// Created by 周梓康 on 2020/3/8.
//

#include "action.h"
#include "state.h"

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
  if (!is) {
    std::cerr << "Error: Invalid input." << std::endl;
  }
}

Action &Action::operator=(Action &&rhs) noexcept {
  pile_id_ = rhs.pile_id_;
  num_objects_ = rhs.num_objects_;
  return *this;
}

bool Action::IsLegal(const State &state) const {
  return pile_id_ >= 0 && pile_id_ <= state.Size() - 1 && num_objects_ >= 1
      && num_objects_ <= state[pile_id_];
}

std::istream &operator>>(std::istream &is, Action &action) {
  Action action_is(is);
  if (is) {
    action = std::move(action_is);
  }
  return is;
}

std::ostream &operator<<(std::ostream &os, const Action &action) {
  os << "From pile " << action.GetPileId() << " remove "
     << action.GetNumObjects() << " object";
  if (action.GetNumObjects() > 1) {
    os << "s";
  }
  return os;
}

bool operator==(const Action &lhs, const Action &rhs) {
  return lhs.GetPileId() == rhs.GetPileId()
      && lhs.GetNumObjects() == rhs.GetNumObjects();
}

bool operator!=(const Action &lhs, const Action &rhs) {
  return !(lhs == rhs);
}
