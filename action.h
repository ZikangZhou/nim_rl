//
// Created by 周梓康 on 2020/3/8.
//

#ifndef NIM_ACTION_H_
#define NIM_ACTION_H_

#include <iostream>
#include <sstream>
#include <string>

class State;

class Action {
 public:
  Action() = default;
  Action(int pile_id, int num_objects)
      : pile_id_(pile_id), num_objects_(num_objects) {}
  explicit Action(std::istream &);
  Action(const Action &) = default;
  Action(Action &&action) noexcept
      : pile_id_(action.pile_id_), num_objects_(action.num_objects_) {}
  Action &operator=(const Action &) = default;
  Action &operator=(Action &&rhs) noexcept;
  ~Action() = default;
  int GetNumObjects() const { return num_objects_; }
  int GetPileId() const { return pile_id_; }
  bool IsLegal(const State &) const;
  void SetNumObjects(int num_object) { num_objects_ = num_object; }
  void SetPileId(int pile_id) { pile_id_ = pile_id; }

 private:
  int pile_id_ = -1;
  int num_objects_ = -1;
};

std::istream &operator>>(std::istream &, Action &);
std::ostream &operator<<(std::ostream &, const Action &);
bool operator==(const Action &, const Action &);
bool operator!=(const Action &, const Action &);

#endif  // NIM_ACTION_H_
