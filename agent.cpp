//
// Created by 周梓康 on 2020/3/3.
//

#include "agent.h"
#include "game.h"

Agent &Agent::operator=(Agent &&rhs) noexcept {
  if (this != &rhs) {
    RemoveFromGames();
    MoveGames(&rhs);
  }
  return *this;
}

Action Agent::SampleAction(const std::vector<Action> &actions) {
  if (actions.empty()) {
    return Action{};
  } else {
    std::uniform_int_distribution<decltype(actions.size())>
        distribution(0, actions.size() - 1);
    return actions[distribution(generator_)];
  }
}

void Agent::MoveGames(Agent *moved_from) {
  games_ = std::move(moved_from->games_);
  for (auto game : games_) {
    if (game) {
      if (game->first_player_ == moved_from) {
        game->first_player_ = this;
      }
      if (game->second_player_ == moved_from) {
        game->second_player_ = this;
      }
    }
  }
  moved_from->games_.clear();
}

void Agent::RemoveFromGames() {
  for (auto game : games_) {
    if (game->first_player_ == this) {
      game->first_player_ = nullptr;
    }
    if (game->second_player_ == this) {
      game->second_player_ = nullptr;
    }
  }
}

Action RandomAgent::Policy(const State &state) {
  return SampleAction(state.ActionSpace());
}

Action HumanAgent::Policy(const State &state) {
  os_ << "Please input two integers to indicate your action." << std::endl;
  Action action;
  while (true) {
    os_ << ">>>";
    if (is_ >> action) {
      if (!state.OutOfRange(action.pile_id()) && action.Valid(state)) {
        break;
      } else {
        os_ << "Invalid action. Please try again." << std::endl;
      }
    } else {
      is_.clear();
    }
  }
  return action;
}

Action OptimalAgent::Policy(const State &state) {
  unsigned nim_sum = 0;
  for (int pile_id = 0; pile_id != state.Size(); ++pile_id) {
    nim_sum ^= state[pile_id];
  }
  for (int pile_id = 0; pile_id != state.Size(); ++pile_id) {
    unsigned num_objects_target = state[pile_id] ^nim_sum;
    if (num_objects_target < state[pile_id]) {
      return Action{pile_id,
                    static_cast<int>(state[pile_id] - num_objects_target)};
    }
  }
  return SampleAction(state.ActionSpace());
}

Action QLearningAgent::Policy(const State &state) {
  std::vector<Action> action_space = state.ActionSpace();
  State end_state(state.Size(), 0);
  if (q_values_.find(end_state) == q_values_.end()) {
    q_values_.insert({end_state, 1.0});
  }
  if (q_values_.find(state) == q_values_.end()) {
    q_values_.insert({state, q_value_distribution(generator)});
  }
  for (const auto &action : action_space) {
    auto key = state.Next(action);
    if (q_values_.find(key) == q_values_.end()) {
      q_values_.insert({key, q_value_distribution(generator)});
    }
  }
  auto iter = std::max_element(action_space.begin(), action_space.end(),
                               [&state, this](const Action &a,
                                              const Action &b) -> bool {
                                 return q_values_[state.Next(a)]
                                     < q_values_[state.Next(b)];
                               });
  if (action_space.empty()) {
    next_state_greedy_ = state;
    return Action{};
  } else {
    next_state_greedy_ = state.Next(*iter);
    if (epsilon_distribution(generator) < epsilon_) {
      return SampleAction(action_space);
    } else {
      return *iter;
    }
  }
}

void QLearningAgent::Update(const State &current_state,
                            const State &next_state,
                            Reward reward) {
  if (next_state.End()) {
    q_values_[current_state] += alpha_
        * (reward - q_values_[current_state]);
  } else {
    q_values_[current_state] += alpha_
        * (reward + gamma_ * q_values_[next_state_greedy_]
            - q_values_[current_state]);
  }
}
