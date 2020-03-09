//
// Created by 周梓康 on 2020/3/8.
//

#ifndef NIM_ACTION_H_
#define NIM_ACTION_H_

#include <iostream>
#include <sstream>
#include <string>
#include <utility>

class Action {
  friend std::istream &operator>>(std::istream &, Action &);
  friend std::ostream &operator<<(std::ostream &, const Action &);

 public:
  Action() : action_(-1, -1) {}

  Action(int pile_id, int num_objects) : action_(pile_id, num_objects) {}

  Action(const Action &) = default;

  Action(Action &&action) noexcept : action_(std::move(action.action_)) {}

  Action &operator=(const Action &) = default;

  Action &operator=(Action &&rhs) noexcept;

  ~Action() = default;

  int &get_pile_id() { return action_.first; }

  const int &get_pile_id() const { return action_.first; }

  int &get_num_objects() { return action_.second; }

  const int &get_num_objects() const { return action_.second; }

  bool Valid(unsigned max_num_objects) const {
    return action_.first >= 0 && action_.second >= 1
        && action_.second <= max_num_objects;
  }

 private:
  std::pair<int, int> action_;
};

std::istream &operator>>(std::istream &, Action &);

std::ostream &operator<<(std::ostream &, const Action &);

#endif  // NIM_ACTION_H_
