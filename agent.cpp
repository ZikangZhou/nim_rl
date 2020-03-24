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

void TDAgent::InitializeQValues(const State &initial_state) {
  DoInitializeQValues(initial_state, 0);
  q_values_[State(initial_state.Size(), 0)] = 1.0;
}

double TDAgent::OptimalActionsRatio() {
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

Action TDAgent::Policy(const State &state) {
  std::vector<Action> legal_actions = state.LegalActions();
  num_legal_actions_ = legal_actions.size();
  if (legal_actions.empty()) {
    greedy_q_ = 0.0;
    return Action{};
  } else {
    Action greedy_action = *std::max_element(legal_actions.begin(),
                                             legal_actions.end(),
                                             [&state, this](const Action &a1,
                                                            const Action &a2) {
                                               return q_values_[state.Child(a1)]
                                                   < q_values_[state.Child(a2)];
                                             });
    greedy_q_ = q_values_[state.Child(greedy_action)];
    std::vector<Action> greedy_actions;
    std::copy_if(legal_actions.begin(),
                 legal_actions.end(),
                 std::back_inserter(greedy_actions),
                 [&state, this](const Action &action) {
                   return q_values_[state.Child(action)] == greedy_q_;
                 });
    num_greedy_actions_ = greedy_actions.size();
    return EpsilonGreedy(legal_actions, greedy_actions);
  }
}

void TDAgent::Reset() {
  Agent::Reset();
  greedy_q_ = 0.0;
  num_legal_actions_ = 0;
  num_greedy_actions_ = 0;
}

void TDAgent::DoInitializeQValues(const State &state, int pile_id) {
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

void QLearningAgent::Update(const State &current_state,
                            const State &next_state,
                            Reward reward) {
  if (next_state.IsTerminal()) {
    q_values_[current_state] += alpha_ * (reward - q_values_[current_state]);
  } else if (!current_state.IsEmpty()) {
    q_values_[current_state] += alpha_
        * (reward + gamma_ * greedy_q_ - q_values_[current_state]);
  }
  current_state_ = next_state;
}

void SarsaAgent::Update(const State &current_state,
                        const State &next_state,
                        Reward reward) {
  if (next_state.IsTerminal()) {
    q_values_[current_state] += alpha_ * (reward - q_values_[current_state]);
  } else if (!current_state.IsEmpty()) {
    q_values_[current_state] += alpha_
        * (reward + gamma_ * q_values_[next_state] - q_values_[current_state]);
  }
  current_state_ = next_state;
}

Action ExpectedSarsaAgent::Policy(const State &state) {
  SetNextStates(state.Children());
  return TDAgent::Policy(state);
}

void ExpectedSarsaAgent::Reset() {
  TDAgent::Reset();
  next_states_.clear();
}

void ExpectedSarsaAgent::Update(const State &current_state,
                                const State &next_state,
                                Reward reward) {
  if (next_state.IsTerminal()) {
    q_values_[current_state] += alpha_ * (reward - q_values_[current_state]);
  } else if (!current_state.IsEmpty()) {
    double expectation = 0.0;
    for (const auto &state : next_states_) {
      if (q_values_[state] != greedy_q_) {
        expectation += epsilon_ / num_legal_actions_ * q_values_[state];
      }
    }
    expectation += (1 - epsilon_) * greedy_q_
        + num_greedy_actions_ * epsilon_ * greedy_q_ / num_legal_actions_;
    q_values_[current_state] +=
        alpha_ * (reward + gamma_ * expectation - q_values_[current_state]);
  }
  current_state_ = next_state;
}

std::unordered_map<State,
                   Agent::Reward> DoubleLearningAgent::GetQValues() const {
  std::unordered_map<State, Reward> q_values(q_values_);
  for (const auto &q_value : q_values_2_) {
    q_values[q_value.first] = (q_values[q_value.first] + q_value.second) / 2;
  }
  return q_values;
}

void DoubleLearningAgent::InitializeQValues(const State &initial_state) {
  TDAgent::InitializeQValues(initial_state);
  q_values_2_ = q_values_;
}

Action DoubleLearningAgent::Policy(const State &state) {
  std::vector<Action> legal_actions = state.LegalActions();
  num_legal_actions_ = legal_actions.size();
  if (legal_actions.empty()) {
    greedy_q_ = 0.0;
    return Action{};
  } else {
    Action greedy_action =
        *std::max_element(legal_actions.begin(),
                          legal_actions.end(),
                          [&state, this](const Action &a1,
                                         const Action &a2) -> bool {
                            State state1 = state.Child(a1);
                            State state2 = state.Child(a2);
                            return q_values_[state1] + q_values_2_[state1]
                                < q_values_[state2] + q_values_2_[state2];
                          });
    State greedy_next_state = state.Child(greedy_action);
    greedy_q_ = q_values_[greedy_next_state] + q_values_2_[greedy_next_state];
    std::vector<Action> greedy_actions;
    std::copy_if(legal_actions.begin(),
                 legal_actions.end(),
                 std::back_inserter(greedy_actions),
                 [&state, this](const Action &action) -> bool {
                   State next_state = state.Child(action);
                   return q_values_[next_state] + q_values_2_[next_state]
                       == greedy_q_;
                 });
    prob_ = dist_(rng_);
    if (prob_ < 0.5) {
      greedy_action = *std::max_element(legal_actions.begin(),
                                        legal_actions.end(),
                                        [&state, this](const Action &a1,
                                                       const Action &a2) {
                                          return q_values_[state.Child(a1)]
                                              < q_values_[state.Child(a2)];
                                        });
      greedy_q_ = q_values_2_[state.Child(greedy_action)];
    } else {
      greedy_action = *std::max_element(legal_actions.begin(),
                                        legal_actions.end(),
                                        [&state, this](const Action &a1,
                                                       const Action &a2) {
                                          return q_values_2_[state.Child(a1)]
                                              < q_values_2_[state.Child(a2)];
                                        });
      greedy_q_ = q_values_[state.Child(greedy_action)];
    }
    num_greedy_actions_ = greedy_actions.size();
    return EpsilonGreedy(legal_actions, greedy_actions);
  }
}

void DoubleLearningAgent::Reset() {
  TDAgent::Reset();
  prob_ = 0.0;
}

void DoubleLearningAgent::Update(const State &current_state,
                                 const State &next_state,
                                 Reward reward) {
  prob_ < 0.5 ? DoUpdate(current_state, next_state, reward, &q_values_)
              : DoUpdate(current_state, next_state, reward, &q_values_2_);
  current_state_ = next_state;
}

void DoubleQLearningAgent::DoUpdate(const State &current_state,
                                    const State &next_state,
                                    Reward reward,
                                    std::unordered_map<State,
                                                       Reward> *q_values) {
  if (next_state.IsTerminal()) {
    (*q_values)[current_state] +=
        alpha_ * (reward - (*q_values)[current_state]);
  } else if (!current_state.IsEmpty()) {
    (*q_values)[current_state] += alpha_
        * (reward + gamma_ * greedy_q_ - (*q_values)[current_state]);
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
