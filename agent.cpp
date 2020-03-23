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
        dist(0, actions.size() - 1);
    return actions[dist(rng_)];
  }
}

void Agent::MoveGames(Agent *moved_from) {
  games_ = std::move(moved_from->games_);
  for (auto &game : games_) {
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
  for (auto &game : games_) {
    if (game->first_player_ == this) {
      game->first_player_ = nullptr;
    }
    if (game->second_player_ == this) {
      game->second_player_ = nullptr;
    }
  }
}

Action RandomAgent::Policy(const State &state) {
  return SampleAction(state.LegalActions());
}

Action HumanAgent::Policy(const State &state) {
  os_ << "Please input two integers to indicate your action." << std::endl;
  Action action;
  while (true) {
    os_ << ">>>";
    if (is_ >> action) {
      if (!state.OutOfRange(action.GetPileId()) && action.IsLegal(state)) {
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
  return SampleAction(state.LegalActions());
}

void QLearningAgent::InitializeQValues(const State &initial_state) {
  DoInitializeQValues(initial_state, 0);
  q_values_[State(initial_state.Size(), 0)] = 1.0;
}

double QLearningAgent::OptimalActionsRatio() {
  double num_n_positions = 0.0;
  double num_optimal_actions = 0.0;
  double initial_epsilon = epsilon_;
  epsilon_ = 0.0;
  for (const auto &q_value : q_values_) {
    if (q_value.first.NimSum()) {
      ++num_n_positions;
      if (!q_value.first.Child(Policy(q_value.first)).NimSum()) {
        ++num_optimal_actions;
      }
    }
  }
  epsilon_ = initial_epsilon;
  return num_optimal_actions / num_n_positions;
}

Action QLearningAgent::Policy(const State &state) {
  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.empty()) {
    next_state_greedy_ = state;
    return Action{};
  } else {
    Action greedy_action = *std::max_element(legal_actions.begin(),
                                             legal_actions.end(),
                                             [&state, this](const Action &a1,
                                                            const Action &a2) {
                                               return q_values_[state.Child(a1)]
                                                   < q_values_[state.Child(a2)];
                                             });
    Reward greedy_q = q_values_[state.Child(greedy_action)];
    std::vector<Action> greedy_actions;
    std::copy_if(legal_actions.begin(),
                 legal_actions.end(),
                 std::back_inserter(greedy_actions),
                 [&state, &greedy_q, this](const Action &action) {
                   return q_values_[state.Child(action)] == greedy_q;
                 });
    greedy_action = SampleAction(greedy_actions);
    next_state_greedy_ = state.Child(greedy_action);
    if (dist_epsilon_(rng_) < epsilon_) {
      return SampleAction(legal_actions);
    } else {
      return greedy_action;
    }
  }
}

void QLearningAgent::Reset() {
  Agent::Reset();
  next_state_greedy_ = State();
}

void QLearningAgent::Update(const State &current_state,
                            const State &next_state,
                            Reward reward) {
  if (next_state.IsTerminal()) {
    q_values_[current_state] += alpha_
        * (reward - q_values_[current_state]);
  } else if (!current_state.IsEmpty()) {
    q_values_[current_state] += alpha_
        * (reward + gamma_ * q_values_[next_state_greedy_]
            - q_values_[current_state]);
  }
  current_state_ = next_state;
}

void QLearningAgent::DoInitializeQValues(const State &state, int pile_id) {
  if (pile_id == state.Size()) {
    if (q_values_.find(state) == q_values_.end()) {
      q_values_.insert({state, dist_q_value_(rng_)});
    }
    return;
  }
  Action action(pile_id, -1);
  DoInitializeQValues(state, pile_id + 1);
  for (int num_objects = 1; num_objects != state[pile_id] + 1; ++num_objects) {
    action.SetNumObjects(num_objects);
    DoInitializeQValues(state.Child(action), pile_id + 1);
  }
}

std::ostream &operator<<(std::ostream &os,
                         const std::unordered_map<State,
                                                  Agent::Reward> &q_values) {
  os << std::fixed << std::setprecision(4);
  for (const auto &q_value : q_values) {
    os << q_value.first << ": " << q_value.second << " ";
  }
  return os;
}
