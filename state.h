//
// Created by 周梓康 on 2020/3/3.
//

#ifndef NIM_STATE_H
#define NIM_STATE_H

#include <stdexcept>
#include <utility>
#include <vector>

class State {
public:
    State() = default;

    explicit State(std::vector<unsigned> state);

    bool empty() const { return state_.empty(); }

    std::vector<unsigned> &get() { return state_; }

    const std::vector<unsigned> &get() const { return state_; }

    unsigned &get(std::vector<unsigned>::size_type pile_id);

    const unsigned &get(std::vector<unsigned>::size_type pile_id) const;

    std::vector<unsigned>::size_type size() const { return state_.size(); }

    void set(std::vector<unsigned>::size_type pile_id, unsigned num_objects);

    void set(const std::vector<unsigned> &state);

    unsigned &operator[](std::vector<unsigned>::size_type pile_id) {
        return state_[pile_id];
    }

    const unsigned &operator[](std::vector<unsigned>::size_type pile_id) const {
        return state_[pile_id];
    }

private:
    std::vector<unsigned> state_{1, 1, 1};
};

#endif //NIM_STATE_H
