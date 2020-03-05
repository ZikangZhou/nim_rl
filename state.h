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

class State {
  friend void swap(State &, State &);

 public:
  typedef std::vector<unsigned>::size_type size_type;

  State() = default;

  explicit State(std::istream &is);

  State(std::initializer_list<unsigned> state);

  explicit State(std::vector<unsigned> state);

  State(const State &) = default;

  State(State &&state) noexcept;

  State &operator=(State state);

  State &operator=(std::initializer_list<unsigned> state);

  State &operator=(const std::vector<unsigned> &state);

  ~State() = default;

  void clear() { state_.clear(); }

  bool empty() const { return state_.empty(); }

  std::vector<unsigned> &get() { return state_; }

  const std::vector<unsigned> &get() const { return state_; }

  void push_back(const unsigned &num_objects) { state_.push_back(num_objects); }

  void RemoveObjects(size_type pile_id,
                     unsigned num_objects);

  size_type size() const { return state_.size(); }

  unsigned &operator[](size_type pile_id);

  const unsigned &operator[](size_type pile_id) const;

 private:
  std::vector<unsigned> state_;

  void Check(size_type pile_id, const std::string &msg) const {
    if (pile_id >= state_.size()) {
      throw std::out_of_range(msg);
    }
  }
};

inline void swap(State &lhs, State &rhs) {
  using std::swap;
  swap(lhs.state_, rhs.state_);
}

std::istream &operator>>(std::istream &is, State &state);

std::ostream &operator<<(std::ostream &os, const State &state);

#endif  // NIM_STATE_H_
