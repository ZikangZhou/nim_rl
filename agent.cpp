//
// Created by 周梓康 on 2020/3/3.
//

#include "agent.h"
#include "game.h"

Agent::Agent(Agent &&agent) noexcept {
  MoveGames(&agent);
}

Agent &Agent::operator=(Agent &&rhs) noexcept {
  if (this != &rhs) {
    RemoveFromGames();
    MoveGames(&rhs);
  }
  return *this;
}

Agent::~Agent() {
  RemoveFromGames();
}

Action Agent::RandomPickAction(const std::vector<Action> &actions) {
  if (actions.empty()) {
    return Action{-1, -1};
  } else {
    std::uniform_int_distribution<decltype(actions.size())>
        distribution(0, actions.size() - 1);
    return actions[distribution(generator)];
  }
}

void Agent::MoveGames(Agent *moved_from) {
  games_ = std::move(moved_from->games_);
  for (auto game : games_) {
    if (game->first_player_ == moved_from) {
      game->first_player_ = this;
    }
    if (game->second_player_ == moved_from) {
      game->second_player_ = this;
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

Action RandomAgent::Policy(Game *game) {
  return RandomPickAction(game->GetActions());
}

Action HumanAgent::Policy(Game *game) {
  os_ << "Please input two integers to indicate your action." << std::endl;
  Action action;
  while (true) {
    os_ << ">>";
    if (is_ >> action) {
      if (!game->get_state().OutOfRange(action.get_pile_id())
          && action.Valid(game->get_state()[action.get_pile_id()])) {
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

Action OptimalAgent::Policy(Game *game) {
  unsigned nim_sum = 0;
  State &state = game->get_state();
  for (State::size_type pile_id = 0; pile_id != state.size(); ++pile_id) {
    nim_sum ^= state[pile_id];
  }
  for (State::size_type pile_id = 0; pile_id != state.size(); ++pile_id) {
    unsigned target_num_objects = state[pile_id] ^nim_sum;
    if (target_num_objects < state[pile_id]) {
      return Action{static_cast<int>(pile_id),
                    static_cast<int>(state[pile_id] - target_num_objects)};
    }
  }
  return RandomPickAction(game->GetActions());
}
