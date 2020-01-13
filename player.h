//
// Created by 周梓康 on 2019/12/26.
//

#ifndef NIM_PLAYER_H
#define NIM_PLAYER_H

#include "position.h"
#include "model.h"

class Player {
private:
    Position position;

public:
    Player(Position pos);
    Position getPosition();
    void Action(Model &model, int pile_id, int num);

};

#endif //NIM_PLAYER_H
