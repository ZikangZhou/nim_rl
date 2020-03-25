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

void ValueIterationAgent::InitializeMaps(const std::vector<State> &all_states) {
  for (const auto &state : all_states) {
    if (state.IsTerminal()) {
      values_.insert({state, 1.0});
    } else {
      values_.insert({state, 0.0});
      std::vector<Action> legal_actions = state.LegalActions();
      for (const auto &action : legal_actions) {
        State next_state = state.Child(action);
        std::vector<StateProb> possibilities{{next_state, 1.0}};
        transitions_.insert({{state, action}, possibilities});
      }
    }
  }
}

Action ValueIterationAgent::Policy(const State &state) {
  std::vector<Action> legal_actions = state.LegalActions();
  return *std::max_element(legal_actions.begin(),
                           legal_actions.end(),
                           [&state, this](const Action &a1,
                                          const Action &a2) {
                             return values_[state.Child(a1)]
                                 < values_[state.Child(a2)];
                           });
}

void ValueIterationAgent::Train(const State &initial_state) {
  std::vector<State> all_states = initial_state.GetAllStates();
  InitializeMaps(all_states);
  double delta;
  do {
    delta = 0.0;
    for (const auto &state : all_states) {
      if (state.IsTerminal()) continue;
      Reward value = 1.0;
      std::vector<Action> legal_actions = state.LegalActions();
      for (const auto &action : legal_actions) {
        auto possibilities = transitions_[{state, action}];
        Reward q_value = 0.0;
        for (const auto &outcome : possibilities) {
          q_value += -outcome.second * values_[outcome.first];
        }
        value = std::min(value, q_value);
      }
      Reward *stored_value = &values_[state];
      delta = std::max(delta, std::abs(*stored_value - value));
      *stored_value = value;
    }
  } while (delta > threshold_);
  std::cout << values_ << std::endl;
}

void TDAgent::InitializeValues(const State &initial_state) {
  DoInitializeValues(initial_state, 0);
  values_[State(initial_state.Size(), 0)] = 1.0;
}

double TDAgent::OptimalActionsRatio() {
  double num_n_positions = 0.0;
  double num_optimal_actions = 0.0;
  double initial_epsilon = epsilon_;
  epsilon_ = 0.0;
  for (const auto &value : values_) {
    if (value.first.NimSum()) {
      ++num_n_positions;
      if (!value.first.Child(Policy(value.first)).NimSum()) {
        ++num_optimal_actions;
      }
    }
  }
  epsilon_ = initial_epsilon;
  return num_optimal_actions / num_n_positions;
}

Action TDAgent::Policy(const State &state) {
  legal_actions_ = state.LegalActions();
  num_legal_actions_ = legal_actions_.size();
  if (legal_actions_.empty()) {
    greedy_value_ = 0.0;
    return Action{};
  } else {
    Action greedy_action = *std::max_element(legal_actions_.begin(),
                                             legal_actions_.end(),
                                             [&state, this](const Action &a1,
                                                            const Action &a2) {
                                               return values_[state.Child(a1)]
                                                   < values_[state.Child(a2)];
                                             });
    greedy_value_ = values_[state.Child(greedy_action)];
    std::vector<Action> greedy_actions;
    std::copy_if(legal_actions_.begin(),
                 legal_actions_.end(),
                 std::back_inserter(greedy_actions),
                 [&state, this](const Action &action) {
                   return values_[state.Child(action)] == greedy_value_;
                 });
    num_greedy_actions_ = greedy_actions.size();
    return EpsilonGreedy(legal_actions_, greedy_actions);
  }
}

void TDAgent::Reset() {
  Agent::Reset();
  greedy_value_ = 0.0;
  legal_actions_.clear();
  num_legal_actions_ = 0;
  num_greedy_actions_ = 0;
}

void TDAgent::DoInitializeValues(const State &state, int pile_id) {
  if (pile_id == state.Size()) {
    if (values_.find(state) == values_.end()) {
      values_.insert({state, dist_value_(rng_)});
    }
    return;
  }
  Action action(pile_id, -1);
  DoInitializeValues(state, pile_id + 1);
  for (int num_objects = 1; num_objects != state[pile_id] + 1; ++num_objects) {
    action.SetNumObjects(num_objects);
    DoInitializeValues(state.Child(action), pile_id + 1);
  }
}

void QLearningAgent::Update(const State &current_state,
                            const State &next_state,
                            Reward reward) {
  if (next_state.IsTerminal()) {
    values_[current_state] += alpha_ * (reward - values_[current_state]);
  } else if (!current_state.IsEmpty()) {
    values_[current_state] += alpha_
        * (reward + gamma_ * greedy_value_ - values_[current_state]);
  }
  current_state_ = next_state;
}

void SarsaAgent::Update(const State &current_state,
                        const State &next_state,
                        Reward reward) {
  if (next_state.IsTerminal()) {
    values_[current_state] += alpha_ * (reward - values_[current_state]);
  } else if (!current_state.IsEmpty()) {
    values_[current_state] += alpha_
        * (reward + gamma_ * values_[next_state] - values_[current_state]);
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
    values_[current_state] += alpha_ * (reward - values_[current_state]);
  } else if (!current_state.IsEmpty()) {
    double expectation = 0.0;
    for (const auto &state : next_states_) {
      if (values_[state] != greedy_value_) {
        expectation += epsilon_ / num_legal_actions_ * values_[state];
      }
    }
    expectation += (1 - epsilon_) * greedy_value_
        + num_greedy_actions_ * epsilon_ * greedy_value_ / num_legal_actions_;
    values_[current_state] +=
        alpha_ * (reward + gamma_ * expectation - values_[current_state]);
  }
  current_state_ = next_state;
}

std::unordered_map<State,
                   Agent::Reward> DoubleLearningAgent::GetValues() const {
  std::unordered_map<State, Reward> values(values_);
  for (const auto &value : values_2_) {
    values[value.first] = (values[value.first] + value.second) / 2;
  }
  return values;
}

void DoubleLearningAgent::InitializeValues(const State &initial_state) {
  TDAgent::InitializeValues(initial_state);
  values_2_ = values_;
}

Action DoubleLearningAgent::Policy(const State &state) {
  legal_actions_ = state.LegalActions();
  num_legal_actions_ = legal_actions_.size();
  if (legal_actions_.empty()) {
    greedy_value_ = 0.0;
    return Action{};
  } else {
    Action greedy_action =
        *std::max_element(legal_actions_.begin(),
                          legal_actions_.end(),
                          [&state, this](const Action &a1,
                                         const Action &a2) -> bool {
                            State state1 = state.Child(a1);
                            State state2 = state.Child(a2);
                            return values_[state1] + values_2_[state1]
                                < values_[state2] + values_2_[state2];
                          });
    State greedy_next_state = state.Child(greedy_action);
    greedy_value_ = values_[greedy_next_state] + values_2_[greedy_next_state];
    std::vector<Action> greedy_actions;
    std::copy_if(legal_actions_.begin(),
                 legal_actions_.end(),
                 std::back_inserter(greedy_actions),
                 [&state, this](const Action &action) -> bool {
                   State next_state = state.Child(action);
                   return values_[next_state] + values_2_[next_state]
                       == greedy_value_;
                 });
    prob_ = dist_(rng_);
    if (prob_ < 0.5) {
      greedy_action = *std::max_element(legal_actions_.begin(),
                                        legal_actions_.end(),
                                        [&state, this](const Action &a1,
                                                       const Action &a2) {
                                          return values_[state.Child(a1)]
                                              < values_[state.Child(a2)];
                                        });
      greedy_value_ = values_2_[state.Child(greedy_action)];
    } else {
      greedy_action = *std::max_element(legal_actions_.begin(),
                                        legal_actions_.end(),
                                        [&state, this](const Action &a1,
                                                       const Action &a2) {
                                          return values_2_[state.Child(a1)]
                                              < values_2_[state.Child(a2)];
                                        });
      greedy_value_ = values_[state.Child(greedy_action)];
    }
    num_greedy_actions_ = greedy_actions.size();
    return EpsilonGreedy(legal_actions_, greedy_actions);
  }
}

void DoubleLearningAgent::Reset() {
  TDAgent::Reset();
  prob_ = 0.0;
}

void DoubleLearningAgent::Update(const State &current_state,
                                 const State &next_state,
                                 Reward reward) {
  prob_ < 0.5 ? DoUpdate(current_state, next_state, reward, &values_)
              : DoUpdate(current_state, next_state, reward, &values_2_);
  current_state_ = next_state;
}

void DoubleQLearningAgent::DoUpdate(const State &current_state,
                                    const State &next_state,
                                    Reward reward,
                                    std::unordered_map<State,
                                                       Reward> *values) {
  if (next_state.IsTerminal()) {
    (*values)[current_state] +=
        alpha_ * (reward - (*values)[current_state]);
  } else if (!current_state.IsEmpty()) {
    (*values)[current_state] += alpha_
        * (reward + gamma_ * greedy_value_ - (*values)[current_state]);
  }
}

void DoubleSarsaAgent::DoUpdate(const State &current_state,
                                const State &next_state,
                                Reward reward,
                                std::unordered_map<State,
                                                   Reward> *values) {
  if (next_state.IsTerminal()) {
    (*values)[current_state] +=
        alpha_ * (reward - (*values)[current_state]);
  } else if (!current_state.IsEmpty()) {
    if (values == &values_) {
      (*values)[current_state] += alpha_
          * (reward + gamma_ * values_2_[next_state]
              - (*values)[current_state]);
    } else {
      (*values)[current_state] += alpha_
          * (reward + gamma_ * values_[next_state]
              - (*values)[current_state]);
    }
  }
}

Action DoubleExpectedSarsaAgent::Policy(const State &state) {
  SetNextStates(state.Children());
  Action action = DoubleLearningAgent::Policy(state);
  Action greedy_action;
  std::vector<Action> greedy_actions;
  if (prob_ < 0.5) {
    greedy_action = *std::max_element(legal_actions_.begin(),
                                      legal_actions_.end(),
                                      [&state, this](const Action &a1,
                                                     const Action &a2) {
                                        return values_2_[state.Child(a1)]
                                            < values_2_[state.Child(a2)];
                                      });
    greedy_value_ = values_2_[state.Child(greedy_action)];
    std::copy_if(legal_actions_.begin(),
                 legal_actions_.end(),
                 std::back_inserter(greedy_actions),
                 [&state, this](const Action &action) {
                   return values_2_[state.Child(action)] == greedy_value_;
                 });
  } else {
    greedy_action = *std::max_element(legal_actions_.begin(),
                                      legal_actions_.end(),
                                      [&state, this](const Action &a1,
                                                     const Action &a2) {
                                        return values_[state.Child(a1)]
                                            < values_[state.Child(a2)];
                                      });
    greedy_value_ = values_[state.Child(greedy_action)];
    std::copy_if(legal_actions_.begin(),
                 legal_actions_.end(),
                 std::back_inserter(greedy_actions),
                 [&state, this](const Action &action) {
                   return values_[state.Child(action)] == greedy_value_;
                 });
  }
  num_greedy_actions_ = greedy_actions.size();
  return action;
}

void DoubleExpectedSarsaAgent::Reset() {
  DoubleLearningAgent::Reset();
  next_states_.clear();
}

void DoubleExpectedSarsaAgent::DoUpdate(const State &current_state,
                                        const State &next_state,
                                        Reward reward,
                                        std::unordered_map<State,
                                                           Reward> *values) {
  if (next_state.IsTerminal()) {
    (*values)[current_state] += alpha_ * (reward - (*values)[current_state]);
  } else if (!current_state.IsEmpty()) {
    double expectation = 0.0;
    if (values == &values_) {
      for (const auto &state : next_states_) {
        if (values_2_[state] != greedy_value_) {
          expectation += epsilon_ / num_legal_actions_ * values_2_[state];
        }
      }
    } else {
      for (const auto &state : next_states_) {
        if (values_[state] != greedy_value_) {
          expectation += epsilon_ / num_legal_actions_ * values_[state];
        }
      }
    }
    expectation += (1 - epsilon_) * greedy_value_
        + num_greedy_actions_ * epsilon_ * greedy_value_ / num_legal_actions_;
    (*values)[current_state] +=
        alpha_ * (reward + gamma_ * expectation - (*values)[current_state]);
  }
}

std::ostream &operator<<(std::ostream &os,
                         const std::unordered_map<State,
                                                  Agent::Reward> &values) {
  os << std::fixed << std::setprecision(4);
  for (const auto &value : values) {
    os << value.first << ": " << value.second << " ";
  }
  return os;
}
