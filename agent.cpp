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

Action Agent::Step(Game *game, bool is_evaluation) {
  Action action = Policy(game->GetState(), is_evaluation);
  game->Step(action);
  return action;
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

Action RandomAgent::Policy(const State &state, bool /*is_evaluation*/) {
  return SampleAction(state.LegalActions());
}

Action HumanAgent::Policy(const State &state, bool /*is_evaluation*/) {
  os_ << "Please input two integers to indicate your action." << std::endl;
  Action action;
  while (true) {
    os_ << ">>>";
    if (is_ >> action) {
      if (!state.OutOfRange(action.GetPileId()) && action.IsLegal(state)) break;
      os_ << "Invalid action. Please try again." << std::endl;
    } else {
      is_.clear();
    }
  }
  return action;
}

Action OptimalAgent::Policy(const State &state, bool /*is_evaluation*/) {
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

void RLAgent::Initialize(const std::vector<State> &all_states) {
  for (const auto &state : all_states) {
    if (state.IsTerminal()) {
      values_.insert({state, kWinReward});
    } else {
      values_.insert({state, kTieReward});
    }
  }
  values_.insert({State(), kTieReward});
}

double RLAgent::OptimalActionsRatio() {
  double num_n_positions = 0.0;
  double num_optimal_actions = 0.0;
  std::unordered_map<State, Reward> values = GetValues();
  for (const auto &value : values) {
    if (value.first.NimSum()) {
      ++num_n_positions;
      if (!value.first.Child(Policy(value.first, true)).NimSum()) {
        ++num_optimal_actions;
      }
    }
  }
  return num_optimal_actions / num_n_positions;
}

void RLAgent::Reset() {
  Agent::Reset();
  greedy_value_ = 0.0;
  legal_actions_.clear();
  greedy_actions_.clear();
}

Action RLAgent::Policy(const State &state, bool is_evaluation) {
  legal_actions_ = state.LegalActions();
  greedy_actions_.clear();
  if (legal_actions_.empty()) {
    greedy_value_ = 0.0;
    return Action{};
  } else {
    Action greedy_action =
        *std::max_element(legal_actions_.begin(), legal_actions_.end(),
                          [&](const Action &a1, const Action &a2) {
                            return values_[state.Child(a1)]
                                < values_[state.Child(a2)];
                          });
    greedy_value_ = values_[state.Child(greedy_action)];
    std::copy_if(legal_actions_.begin(), legal_actions_.end(),
                 std::back_inserter(greedy_actions_),
                 [&](const Action &action) {
                   return values_[state.Child(action)] == greedy_value_;
                 });
    if (is_evaluation) {
      return SampleAction(greedy_actions_);
    } else {
      return PolicyImpl(legal_actions_, greedy_actions_);
    }
  }
}

void
ValueIterationAgent::Initialize(const std::vector<State> &all_states) {
  for (const auto &state : all_states) {
    if (state.IsTerminal()) {
      values_.insert({state, kWinReward});
    } else {
      values_.insert({state, kTieReward});
      std::vector<Action> legal_actions = state.LegalActions();
      for (const auto &action : legal_actions) {
        State next_state = state.Child(action);
        std::vector<StateProb> possibilities{{next_state, 1.0}};
        transitions_.insert({{state, action}, possibilities});
      }
    }
  }
  ValueIteration(all_states);
}

void ValueIterationAgent::ValueIteration(const std::vector<State> &all_states) {
  int step = 0;
  double delta;
  do {
    delta = 0.0;
    for (const auto &state : all_states) {
      if (state.IsTerminal()) continue;
      Reward value = kMaxValue;
      std::vector<Action> legal_actions = state.LegalActions();
      for (const auto &action : legal_actions) {
        auto possibilities = transitions_[{state, action}];
        Reward q_value = 0.0;
        for (const auto &outcome : possibilities) {
          q_value += outcome.second * -values_[outcome.first];
        }
        value = std::min(value, q_value);
      }
      Reward *stored_value = &values_[state];
      delta = std::max(delta, std::abs(*stored_value - value));
      *stored_value = value;
    }
    std::cout << "Epoch " << ++step
              << ": Value Iteration agent optimal actions ratio: "
              << OptimalActionsRatio() << std::endl;
  } while (delta > threshold_);
}

void MonteCarloAgent::Initialize(const std::vector<State> &all_states) {
  RLAgent::Initialize(all_states);
  for (const auto &kv : values_) {
    cumulative_sums_.insert({kv.first, 0.0});
  }
}

void MonteCarloAgent::Reset() {
  RLAgent::Reset();
  trajectory_.clear();
}

Action MonteCarloAgent::Step(Game *game, bool is_evaluation) {
  if (!trajectory_.empty()) {
    std::get<2>(trajectory_.back()) = -game->GetReward();
  }
  State state = game->GetState();
  Action action = Agent::Step(game, is_evaluation);
  trajectory_.emplace_back(state, action, game->GetReward());
  if (game->GetState().IsTerminal() || game->GetState().IsEmpty()) {
    Update();
  }
  return action;
}

void MonteCarloAgent::Update() {
  double ret = 0.0;
  for (auto r_iter = trajectory_.crbegin(); r_iter != trajectory_.crend();
       ++r_iter) {
    const State &state = std::get<0>(*r_iter);
    if (!state.IsTerminal()) {
      const Action &action = std::get<1>(*r_iter);
      const State &next_state = state.Child(action);
      ret = gamma_ * ret + std::get<2>(*r_iter);
      if (std::find_if(r_iter + 1, trajectory_.crend(),
                       [&](const TimeStep &time_step) {
                         return std::get<0>(time_step).Child(action)
                             == next_state;
                       }) == trajectory_.crend()) {
        ++cumulative_sums_[next_state];
        values_[next_state] +=
            (ret - values_[next_state]) / cumulative_sums_[next_state];
      }
    }
  }
}

Action ESMonteCarloAgent::Step(Game *game, bool is_evaluation) {
  if (!is_evaluation && game->GetState() == game->GetInitialState()) {
    State start_state = SampleState(game->GetAllStates());
    Action start_action = SampleAction(start_state.LegalActions());
    game->SetState(start_state);
    game->Step(start_action);
    trajectory_.emplace_back(start_state, start_action, game->GetReward());
    return start_action;
  } else {
    return MonteCarloAgent::Step(game, is_evaluation);
  }
}

void OffPolicyMonteCarloAgent::Update() {
  double ret = 0.0, weight = 1.0;
  std::unordered_map<State, Reward> behavior_policy_values = values_;
  for (auto r_iter = trajectory_.crbegin(); r_iter != trajectory_.crend();
       ++r_iter) {
    const State &state = std::get<0>(*r_iter);
    if (state.IsTerminal()) continue;
    const Action &action = std::get<1>(*r_iter);
    const State &next_state = state.Child(action);
    ret = gamma_ * ret + std::get<2>(*r_iter);
    if (importance_sampling_ == ImportanceSampling::kNormal) {
      ++cumulative_sums_[next_state];
      values_[next_state] +=
          (weight * ret - values_[next_state]) / cumulative_sums_[next_state];
    } else if (importance_sampling_ == ImportanceSampling::kWeighted) {
      cumulative_sums_[next_state] += weight;
      values_[next_state] +=
          weight * (ret - values_[next_state]) / cumulative_sums_[next_state];
    }
    if (action != Policy(state, true)) break;
    double behavior_policy_value = behavior_policy_values[state.Child(action)];
    std::vector<Action> legal_actions = state.LegalActions();
    Action behavior_policy_greedy_action =
        *std::max_element(legal_actions.begin(), legal_actions.end(),
                          [&](const Action &a1, const Action &a2) {
                            return behavior_policy_values[state.Child(a1)]
                                < behavior_policy_values[state.Child(a2)];
                          });
    double behavior_policy_greedy_value =
        behavior_policy_values[state.Child(behavior_policy_greedy_action)];
    long num_behavior_policy_greedy_actions =
        std::count_if(legal_actions.begin(), legal_actions.end(),
                      [&](const Action &action) {
                        return behavior_policy_values[state.Child(action)]
                            == behavior_policy_greedy_value;
                      });
    if (behavior_policy_value == behavior_policy_greedy_value) {
      weight /= (1 - epsilon_) / num_behavior_policy_greedy_actions
          + epsilon_ / legal_actions.size();
    } else {
      weight /= epsilon_ / legal_actions.size();
    }
  }
}

Action TDAgent::Step(Game *game, bool is_evaluation) {
  Reward reward = game->GetReward();
  Action action = Agent::Step(game, is_evaluation);
  if (!is_evaluation) {
    Update(current_state_, game->GetState(), -reward);
  }
  return action;
}

void QLearningAgent::Update(const State &update_state,
                            const State &current_state,
                            Reward reward) {
  if (!update_state.IsEmpty()) {
    values_[update_state] +=
        alpha_ * (reward + gamma_ * greedy_value_ - values_[update_state]);
  }
  current_state_ = current_state;
}

void SarsaAgent::Update(const State &update_state,
                        const State &current_state,
                        Reward reward) {
  if (!update_state.IsEmpty()) {
    values_[update_state] += alpha_
        * (reward + gamma_ * values_[current_state] - values_[update_state]);
  }
  current_state_ = current_state;
}

void ExpectedSarsaAgent::Reset() {
  TDAgent::Reset();
  next_states_.clear();
}

void ExpectedSarsaAgent::Update(const State &update_state,
                                const State &current_state,
                                Reward reward) {
  if (!update_state.IsEmpty()) {
    double expectation = 0.0;
    if (!legal_actions_.empty()) {
      for (const auto &state : next_states_) {
        if (values_[state] != greedy_value_) {
          expectation += epsilon_ * values_[state] / legal_actions_.size();
        }
      }
      expectation += (1 - epsilon_) * greedy_value_
          + greedy_actions_.size() * epsilon_ * greedy_value_
              / legal_actions_.size();
    }
    values_[update_state] +=
        alpha_ * (reward + gamma_ * expectation - values_[update_state]);
  }
  current_state_ = current_state;
}

Action ExpectedSarsaAgent::Policy(const State &state, bool is_evaluation) {
  SetNextStates(state.Children());
  return TDAgent::Policy(state, is_evaluation);
}

std::unordered_map<State,
                   Agent::Reward> DoubleLearningAgent::GetValues() const {
  std::unordered_map<State, Reward> values(values_);
  for (const auto &kv : values_2_) {
    values[kv.first] = (values[kv.first] + kv.second) / 2;
  }
  return values;
}

void
DoubleLearningAgent::Initialize(const std::vector<State> &all_states) {
  TDAgent::Initialize(all_states);
  values_2_ = values_;
}

void DoubleLearningAgent::Reset() {
  TDAgent::Reset();
  flag_ = false;
}

Action DoubleLearningAgent::Policy(const State &state, bool is_evaluation) {
  legal_actions_ = state.LegalActions();
  greedy_actions_.clear();
  if (legal_actions_.empty()) {
    greedy_value_ = 0.0;
    return Action{};
  } else {
    Action greedy_action =
        *std::max_element(legal_actions_.begin(), legal_actions_.end(),
                          [&](const Action &a1, const Action &a2) -> bool {
                            State state1 = state.Child(a1);
                            State state2 = state.Child(a2);
                            return values_[state1] + values_2_[state1]
                                < values_[state2] + values_2_[state2];
                          });
    State greedy_state = state.Child(greedy_action);
    greedy_value_ = values_[greedy_state] + values_2_[greedy_state];
    std::copy_if(legal_actions_.begin(), legal_actions_.end(),
                 std::back_inserter(greedy_actions_),
                 [&state, this](const Action &action) -> bool {
                   State next_state = state.Child(action);
                   return values_[next_state] + values_2_[next_state]
                       == greedy_value_;
                 });
    flag_ = dist_flag_(rng_);
    if (flag_) {
      greedy_action =
          *std::max_element(legal_actions_.begin(), legal_actions_.end(),
                            [&](const Action &a1, const Action &a2) {
                              return values_[state.Child(a1)]
                                  < values_[state.Child(a2)];
                            });
      greedy_value_ = values_2_[state.Child(greedy_action)];
    } else {
      greedy_action =
          *std::max_element(legal_actions_.begin(), legal_actions_.end(),
                            [&](const Action &a1, const Action &a2) {
                              return values_2_[state.Child(a1)]
                                  < values_2_[state.Child(a2)];
                            });
      greedy_value_ = values_[state.Child(greedy_action)];
    }
    if (is_evaluation) {
      return SampleAction(greedy_actions_);
    } else {
      return PolicyImpl(legal_actions_, greedy_actions_);
    }
  }
}

void DoubleLearningAgent::Update(const State &update_state,
                                 const State &current_state,
                                 Reward reward) {
  flag_ ? DoUpdate(update_state, current_state, reward, &values_)
        : DoUpdate(update_state, current_state, reward, &values_2_);
  current_state_ = current_state;
}

void DoubleQLearningAgent::DoUpdate(const State &update_state,
                                    const State &/*current_state*/,
                                    Reward reward,
                                    std::unordered_map<State,
                                                       Reward> *values) {
  if (!update_state.IsEmpty()) {
    (*values)[update_state] += alpha_
        * (reward + gamma_ * greedy_value_ - (*values)[update_state]);
  }
}

void DoubleSarsaAgent::DoUpdate(const State &update_state,
                                const State &current_state,
                                Reward reward,
                                std::unordered_map<State,
                                                   Reward> *values) {
  if (!update_state.IsEmpty()) {
    if (values == &values_) {
      (*values)[update_state] += alpha_
          * (reward + gamma_ * values_2_[current_state]
              - (*values)[update_state]);
    } else {
      (*values)[update_state] += alpha_
          * (reward + gamma_ * values_[current_state]
              - (*values)[update_state]);
    }
  }
}

void DoubleExpectedSarsaAgent::Reset() {
  DoubleLearningAgent::Reset();
  next_states_.clear();
}

void DoubleExpectedSarsaAgent::DoUpdate(const State &update_state,
                                        const State &/*current_state*/,
                                        Reward reward,
                                        std::unordered_map<State,
                                                           Reward> *values) {
  if (!update_state.IsEmpty()) {
    double expectation = 0.0;
    if (!legal_actions_.empty()) {
      if (values == &values_) {
        for (const auto &state : next_states_) {
          if (values_2_[state] != greedy_value_) {
            expectation += values_2_[state] * epsilon_ / legal_actions_.size();
          }
        }
      } else if (values == &values_2_) {
        for (const auto &state : next_states_) {
          if (values_[state] != greedy_value_) {
            expectation += values_[state] * epsilon_ / legal_actions_.size();
          }
        }
      }
      expectation += (1 - epsilon_) * greedy_value_
          + greedy_actions_.size() * epsilon_ * greedy_value_
              / legal_actions_.size();
    }
    (*values)[update_state] +=
        alpha_ * (reward + gamma_ * expectation - (*values)[update_state]);
  }
}

Action DoubleExpectedSarsaAgent::Policy(const State &state,
                                        bool is_evaluation) {
  SetNextStates(state.Children());
  Action action = DoubleLearningAgent::Policy(state, is_evaluation);
  greedy_actions_.clear();
  if (!legal_actions_.empty()) {
    Action greedy_action;
    if (flag_) {
      greedy_action =
          *std::max_element(legal_actions_.begin(), legal_actions_.end(),
                            [&](const Action &a1, const Action &a2) {
                              return values_2_[state.Child(a1)]
                                  < values_2_[state.Child(a2)];
                            });
      greedy_value_ = values_2_[state.Child(greedy_action)];
      std::copy_if(legal_actions_.begin(), legal_actions_.end(),
                   std::back_inserter(greedy_actions_),
                   [&](const Action &action) {
                     return values_2_[state.Child(action)] == greedy_value_;
                   });
    } else {
      greedy_action =
          *std::max_element(legal_actions_.begin(), legal_actions_.end(),
                            [&](const Action &a1, const Action &a2) {
                              return values_[state.Child(a1)]
                                  < values_[state.Child(a2)];
                            });
      greedy_value_ = values_[state.Child(greedy_action)];
      std::copy_if(legal_actions_.begin(), legal_actions_.end(),
                   std::back_inserter(greedy_actions_),
                   [&](const Action &action) {
                     return values_[state.Child(action)] == greedy_value_;
                   });
    }
  }
  return action;
}

void NStepBootstrappingAgent::Reset() {
  TDAgent::Reset();
  current_time_ = 0;
  terminal_time_ = INT_MAX - 1;
  update_time_ = 0;
  trajectory_.clear();
}

Action NStepBootstrappingAgent::Step(Game *game, bool is_evaluation) {
  Action action;
  if (!trajectory_.empty()) {
    std::get<2>(trajectory_.back()) = -game->GetReward();
  }
  State state = game->GetState();
  action = Policy(state, is_evaluation);
  game->Step(action);
  trajectory_.emplace_back(state, action, game->GetReward());
  if (game->GetState().IsTerminal() || game->GetState().IsEmpty()) {
    terminal_time_ = current_time_ + 1;
  }
  if (!is_evaluation) {
    update_time_ = current_time_ - n_;
    if (update_time_ >= 0) {
      TimeStep time_step = trajectory_[update_time_];
      State update_state =
          std::get<0>(time_step).Child(std::get<1>(time_step));
      Update(update_state, game->GetState());
    }
    if (game->GetState().IsTerminal() || game->GetState().IsEmpty()) {
      while (++update_time_ < terminal_time_ - 1) {
        if (update_time_ >= 0) {
          TimeStep time_step = trajectory_[update_time_];
          State update_state =
              std::get<0>(time_step).Child(std::get<1>(time_step));
          Update(update_state, game->GetState());
        }
      }
    }
  }
  ++current_time_;
  return action;
}

void NStepSarsaAgent::Update(const State &update_state,
                             const State &current_state) {
  double ret = 0.0;
  for (int i = update_time_ + 1;
       i < std::min(update_time_ + n_ + 1, terminal_time_ + 1); ++i) {
    if (!std::get<0>(trajectory_[i - 1]).IsTerminal()) {
      ret +=
          pow(gamma_, i - update_time_ - 1) * std::get<2>(trajectory_[i - 1]);
    }
  }
  if (update_time_ + n_ < terminal_time_) {
    ret += pow(gamma_, n_) * values_[current_state];
  }
  values_[update_state] += alpha_ * (ret - values_[update_state]);
}

void NStepExpectedSarsaAgent::Reset() {
  NStepBootstrappingAgent::Reset();
  next_states_.clear();
}

Action NStepExpectedSarsaAgent::Policy(const State &state, bool is_evaluation) {
  SetNextStates(state.Children());
  return NStepBootstrappingAgent::Policy(state, is_evaluation);
}

void NStepExpectedSarsaAgent::Update(const State &update_state,
                                     const State &/*current_state*/) {
  double ret = 0.0;
  for (int i = update_time_ + 1;
       i != std::min(update_time_ + n_ + 1, terminal_time_ + 1); ++i) {
    if (!std::get<0>(trajectory_[i - 1]).IsTerminal()) {
      ret +=
          pow(gamma_, i - update_time_ - 1) * std::get<2>(trajectory_[i - 1]);
    }
  }
  if (update_time_ + n_ < terminal_time_) {
    double expectation = 0.0;
    if (!legal_actions_.empty()) {
      for (const auto &state : next_states_) {
        if (values_[state] != greedy_value_) {
          expectation += epsilon_ * values_[state] / legal_actions_.size();
        }
      }
      expectation += (1 - epsilon_) * greedy_value_
          + greedy_actions_.size() * epsilon_ * greedy_value_
              / legal_actions_.size();
    }
    ret += pow(gamma_, n_) * expectation;
  }
  values_[update_state] += alpha_ * (ret - values_[update_state]);
}

std::ostream &operator<<(std::ostream &os,
                         const std::unordered_map<State,
                                                  Agent::Reward> &values) {
  os << std::fixed << std::setprecision(kPrecision);
  for (const auto &value : values) {
    os << value.first << ": " << value.second << " ";
  }
  return os;
}

std::ostream &operator<<(std::ostream &os,
                         const std::vector<RLAgent::TimeStep> &trajectory) {
  for (const auto &time_step : trajectory) {
    os << std::get<0>(time_step) << ", " << std::get<1>(time_step) << ", "
       << std::get<2>(time_step) << "; ";
  }
  return os;
}

Action SampleAction(const std::vector<Action> &actions) {
  static std::mt19937 rng{std::random_device{}()};
  if (actions.empty()) {
    return Action{};
  } else {
    std::uniform_int_distribution<decltype(actions.size())>
        dist(0, actions.size() - 1);
    return actions[dist(rng)];
  }
}

State SampleState(const std::vector<State> &states) {
  static std::mt19937 rng{std::random_device{}()};
  if (states.empty()) {
    return State{};
  } else {
    std::uniform_int_distribution<decltype(states.size())>
        dist(0, states.size() - 1);
    return states[dist(rng)];
  }
}
