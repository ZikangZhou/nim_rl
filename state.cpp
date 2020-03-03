//
// Created by 周梓康 on 2020/3/3.
//

#include "state.h"

State::State(std::vector<unsigned> state) : state_(std::move(state)) {
    if (state_.empty()) {
        throw std::invalid_argument("State should not be empty");
    }
}

unsigned &State::get(std::vector<unsigned>::size_type pile_id) {
    if (pile_id >= state_.size()) {
        throw std::out_of_range("pile_id should not be out of range");
    }
    return state_[pile_id];
}

const unsigned &State::get(std::vector<unsigned>::size_type pile_id) const {
    if (pile_id >= state_.size()) {
        throw std::out_of_range("pile_id should not be out of range");
    }
    return state_[pile_id];
}

void State::set(std::vector<unsigned>::size_type pile_id, unsigned num_objects) {
    if (pile_id >= state_.size()) {
        throw std::out_of_range("pile_id should not be out of range");
    }
    state_[pile_id] = num_objects;
}

void State::set(const std::vector<unsigned> &state) {
    if (state.empty()) {
        throw std::invalid_argument("State should not be empty");
    }
    state_ = state;
}
