//
// Created by 周梓康 on 2020/3/8.
//

#include "action.h"

Action::Action(Action &&action) noexcept : action_(std::move(action.action_)) {
  action.action_ = std::pair<State::size_type, unsigned>();
}

Action &Action::operator=(Action &&rhs) noexcept {
  action_ = std::move(rhs.action_);
  return *this;
}
