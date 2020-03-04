//
// Created by 周梓康 on 2020/3/3.
//

#ifndef NIM_STATE_H_
#define NIM_STATE_H_

#include <exception>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class State {
 public:
  State() = default;

  explicit State(std::istream &is);

  State(std::initializer_list<unsigned> state);

  explicit State(std::vector<unsigned> state);

  State(const State &) = default;

  State &operator=(const State &) = default;

  State &operator=(std::initializer_list<unsigned> state);

  State &operator=(const std::vector<unsigned> &state);

  ~State() = default;

  void clear() { state_.clear(); }

  bool empty() const { return state_.empty(); }

  std::vector<unsigned> &get() { return state_; }

  const std::vector<unsigned> &get() const { return state_; }

  void push_back(unsigned num_objects) { state_.push_back(num_objects); }

  void RemoveObjects(std::vector<unsigned>::size_type pile_id,
                     unsigned num_objects);

  std::vector<unsigned>::size_type size() const { return state_.size(); }

  unsigned &operator[](std::vector<unsigned>::size_type pile_id);

  const unsigned &operator[](std::vector<unsigned>::size_type pile_id) const;

 private:
  std::vector<unsigned> state_;
};

std::istream &operator>>(std::istream &is, State &state);

std::ostream &operator<<(std::ostream &os, const State &state);

#endif  // NIM_STATE_H_
