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

void Agent::Reset() {
  current_state_ = next_state_greedy_ = State();
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
  unsigned nim_sum = state.NimSum();
  for (int pile_id = 0; pile_id != state.Size(); ++pile_id) {
    unsigned num_objects_target = state[pile_id] ^nim_sum;
    if (num_objects_target < state[pile_id]) {
      return Action{pile_id,
                    static_cast<int>(state[pile_id] - num_objects_target)};
    }
  }
  return SampleAction(state.ActionSpace());
}

double QLearningAgent::ConvergenceRate() {
  double num_n_position = 0.0;
  double num_optimal_action = 0.0;
  double init_epsilon = epsilon_;
  epsilon_ = 0.0;
  for (const auto &q_value : q_values_) {
    if (q_value.first.NimSum()) {
      ++num_n_position;
      if (!q_value.first.Next(Policy(q_value.first)).NimSum()) {
        ++num_optimal_action;
      }
    }
  }
  epsilon_ = init_epsilon;
  return num_optimal_action / num_n_position;
}

void QLearningAgent::InitQValues(const State &init_state) {
  InitQValuesImpl(init_state, 0);
  q_values_[State(init_state.Size(), 0)] = 1.0;
}

Action QLearningAgent::Policy(const State &state) {
  std::vector<Action> action_space = state.ActionSpace();
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
    if (epsilon_distribution_(generator_) < epsilon_) {
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

void QLearningAgent::InitQValuesImpl(const State &state, int pile_id) {
  if (pile_id == state.Size()) {
    if (q_values_.find(state) == q_values_.end()) {
      q_values_.insert({state, q_value_distribution_(generator_)});
    }
    return;
  }
  Action action(pile_id, -1);
  InitQValuesImpl(state, pile_id + 1);
  for (int num_objects = 1; num_objects != state[pile_id] + 1; ++num_objects) {
    action.set_num_objects(num_objects);
    InitQValuesImpl(state.Next(action), pile_id + 1);
  }
}

std::ostream &operator<<(std::ostream &os,
                         const std::unordered_map<State,
                                                  Agent::Reward> &q_values) {
  os << std::fixed << std::setprecision(4);
  for (const auto &q_value : q_values) {
    os << q_value.first << ": " << q_value.second << ", ";
  }
  return os;
}
