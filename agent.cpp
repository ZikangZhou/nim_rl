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

Action Agent::RandomPickAction(const std::vector<Action> &actions) {
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

Action RandomAgent::Policy(Game *game) {
  if (game) {
    return RandomPickAction(game->GetActions());
  } else {
    return Action{};
  }
}

Action HumanAgent::Policy(Game *game) {
  if (game) {
    os_ << "Please input two integers to indicate your action." << std::endl;
    Action action;
    while (true) {
      os_ << ">>";
      if (is_ >> action) {
        if (!game->state().OutOfRange(action.pile_id())
            && action.Valid(game->state())) {
          break;
        } else {
          os_ << "Invalid action. Please try again." << std::endl;
        }
      } else {
        is_.clear();
      }
    }
    return action;
  } else {
    return Action{};
  }
}

Action OptimalAgent::Policy(Game *game) {
  if (game) {
    unsigned nim_sum = 0;
    State &state = game->state();
    for (State::size_type pile_id = 0; pile_id != state.Size(); ++pile_id) {
      nim_sum ^= state[pile_id];
    }
    for (State::size_type pile_id = 0; pile_id != state.Size(); ++pile_id) {
      unsigned num_objects_target = state[pile_id] ^nim_sum;
      if (num_objects_target < state[pile_id]) {
        return Action{static_cast<int>(pile_id),
                      static_cast<int>(state[pile_id] - num_objects_target)};
      }
    }
    return RandomPickAction(game->GetActions());
  } else {
    return Action{};
  }
}
