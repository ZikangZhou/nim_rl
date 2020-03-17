//
// Created by 周梓康 on 2020/3/8.
//

#ifndef NIM_ACTION_H_
#define NIM_ACTION_H_

#include <iostream>
#include <sstream>
#include <string>
#include <utility>

class State;

class Action {
  friend std::ostream &operator<<(std::ostream &, const Action &);
  friend bool operator==(const Action &, const Action &);

 public:
  Action() = default;
  Action(int pile_id, int num_objects)
      : pile_id_(pile_id), num_objects_(num_objects) {}
  explicit Action(std::istream &);
  Action(const Action &) = default;
  Action(Action &&action) noexcept
      : pile_id_(action.pile_id_),
        num_objects_(action.num_objects_) {}
  Action &operator=(const Action &) = default;
  Action &operator=(Action &&rhs) noexcept;
  ~Action() = default;
  int num_objects() { return num_objects_; }
  const int num_objects() const { return num_objects_; }
  int pile_id() { return pile_id_; }
  const int pile_id() const { return pile_id_; }
  void set_num_objects(int num_object) { num_objects_ = num_object; }
  void set_pile_id(int pile_id) { pile_id_ = pile_id; }
  bool Valid(const State &) const;

 private:
  int pile_id_ = -1;
  int num_objects_ = -1;
};

std::istream &operator>>(std::istream &, Action &);
std::ostream &operator<<(std::ostream &, const Action &);
bool operator==(const Action &, const Action &);
bool operator!=(const Action &, const Action &);

#endif  // NIM_ACTION_H_
