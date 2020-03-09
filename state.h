//
// Created by 周梓康 on 2020/3/3.
//

#ifndef NIM_STATE_H_
#define NIM_STATE_H_

#include <initializer_list>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "action.h"

class State {
  friend void swap(State &, State &);
  friend bool operator==(const State &, const State &);

 public:
  using size_type = std::vector<unsigned>::size_type;
  State() = default;
  explicit State(std::istream &);
  State(std::initializer_list<unsigned> il) : state_(il) { CheckEmpty(); }
  explicit State(std::vector<unsigned> vec) : state_(std::move(vec)) {
    CheckEmpty();
  }
  State(const State &state) : state_(state.state_) { CheckEmpty(); }
  State(State &&state) noexcept : state_(std::move(state.state_)) {
    CheckEmpty();
  }
  State &operator=(const State &);
  State &operator=(State &&) noexcept;
  State &operator=(std::initializer_list<unsigned>);
  State &operator=(std::vector<unsigned>);
  ~State() = default;
  void Clear() { state_.clear(); }
  bool Empty() const { return state_.empty(); }
  bool OutOfRange(int pile_id) const {
    return pile_id >= state_.size() || pile_id < 0;
  }
  size_type Size() const { return state_.size(); }
  void TakeAction(const Action &);
  unsigned &operator[](int);
  const unsigned &operator[](int) const;

 private:
  std::vector<unsigned> state_;
  void CheckEmpty(const std::string &msg = "Warning: State is empty.");
  void CheckRange(int pile_id,
                  const std::string &msg = "Pile_id is out of range.") const;
};

std::istream &operator>>(std::istream &, State &);
std::ostream &operator<<(std::ostream &, const State &);
bool operator==(const State &, const State &);
bool operator!=(const State &, const State &);

inline void State::CheckEmpty(const std::string &msg) {
  if (state_.empty()) {
    std::cerr << msg << std::endl;
  }
}

inline void State::CheckRange(int pile_id, const std::string &msg) const {
  if (pile_id >= state_.size() || pile_id < 0) {
    throw std::out_of_range(msg);
  }
}

inline void swap(State &lhs, State &rhs) {
  using std::swap;
  swap(lhs.state_, rhs.state_);
}

#endif  // NIM_STATE_H_
