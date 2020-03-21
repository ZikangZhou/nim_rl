//
// Created by 周梓康 on 2020/3/3.
//

#ifndef NIM_STATE_H_
#define NIM_STATE_H_

#include <algorithm>
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
  friend class std::hash<State>;

 public:
  using size_type = std::vector<unsigned>::size_type;
  State() = default;
  State(size_type n, unsigned val) : data_(n, val) {}
  explicit State(std::istream &);
  State(std::initializer_list<unsigned> il) : data_(il) {}
  explicit State(std::vector<unsigned> vec) : data_(std::move(vec)) {}
  State(const State &state) = default;
  State(State &&state) noexcept : data_(std::move(state.data_)) {}
  State &operator=(const State &) = default;
  State &operator=(State &&) noexcept;
  State &operator=(std::initializer_list<unsigned>);
  State &operator=(std::vector<unsigned>);
  ~State() = default;
  std::vector<Action> ActionSpace() const;
  void Clear() { data_.clear(); }
  bool Empty() const { return data_.empty(); }
  bool End() const;
  State Next(const Action &action) const;
  unsigned NimSum() const;
  bool OutOfRange(int pile_id) const {
    return pile_id >= data_.size() || pile_id < 0;
  }
  size_type Size() const { return data_.size(); }
  void Update(const Action &);
  unsigned &operator[](int);
  const unsigned &operator[](int) const;

 private:
  std::vector<unsigned> data_;
  void CheckEmpty(const std::string &msg = "Warning: State is empty.");
  void CheckRange(int pile_id,
                  const std::string &msg = "Pile_id is out of range.") const;
};

std::istream &operator>>(std::istream &, State &);
std::ostream &operator<<(std::ostream &, const State &);
bool operator==(const State &, const State &);
bool operator!=(const State &, const State &);

inline void State::CheckEmpty(const std::string &msg) {
  if (data_.empty()) {
    std::cerr << msg << std::endl;
  }
}

inline void State::CheckRange(int pile_id, const std::string &msg) const {
  if (pile_id >= data_.size() || pile_id < 0) {
    throw std::out_of_range(msg);
  }
}

inline void swap(State &lhs, State &rhs) {
  using std::swap;
  swap(lhs.data_, rhs.data_);
}

namespace std {
template<>
struct hash<State> {
  typedef std::size_t result_type;
  typedef State argument_type;
  std::size_t operator()(const State &state) const {
    std::vector<unsigned> state_sorted(state.data_);
    std::sort(state_sorted.begin(), state_sorted.end());
    size_t seed = 0;
    for (int pile_id = 0; pile_id != state_sorted.size(); ++pile_id) {
      seed ^= std::hash<unsigned>()(state_sorted[pile_id]) + 0x9e3779b9
          + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};
}  // namespace std

#endif  // NIM_STATE_H_
