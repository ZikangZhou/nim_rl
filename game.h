//
// Created by 周梓康 on 2020/3/2.
//

#ifndef NIM_GAME_H
#define NIM_GAME_H

#include "player.h"
#include "state.h"

class Game {
public:
    Game() = default;

    Game(State state, Player *first_player, Player *second_player);

    bool GameOver() const;

    std::vector<unsigned> &get_state() { return state_.get(); }

    const std::vector<unsigned> &get_state() const { return state_.get(); }

    unsigned &get_state(std::vector<unsigned>::size_type pile_id) { return state_.get(pile_id); }

    const unsigned &get_state(std::vector<unsigned>::size_type pile_id) const { return state_.get(pile_id); }

    void set_first_player(Player *first_player) { first_player_ = first_player; };

    void set_second_player(Player *second_player) { second_player_ = second_player; };

    void set_state(std::vector<unsigned>::size_type pile_id, unsigned num_objects) { state_.set(pile_id, num_objects); }

    void set_state(const std::vector<unsigned> &state) { state_.set(state); }

    void Run();

private:
    State state_;
    Player *first_player_ = nullptr;
    Player *second_player_ = nullptr;
};

#endif //NIM_GAME_H
