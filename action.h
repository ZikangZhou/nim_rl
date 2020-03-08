//
// Created by 周梓康 on 2020/3/8.
//

#ifndef NIM_ACTION_H_
#define NIM_ACTION_H_

#include <utility>

#include "state.h"

class Action {
 public:
  Action(State::size_type pile_id, unsigned num_objects)
      : action_(pile_id, num_objects) {}

  Action(const Action &) = default;

  Action(Action &&) noexcept;

  Action &operator=(const Action &) = default;

  Action &operator=(Action &&rhs) noexcept;

  ~Action() = default;

 private:
  std::pair<State::size_type, unsigned> action_;
};

#endif  // NIM_ACTION_H_
