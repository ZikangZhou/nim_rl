//
// Created by 周梓康 on 2019/12/26.
//

#ifndef NIM_MODEL_H
#define NIM_MODEL_H

#include <vector>
#include "player.h"

class Model {
private:
    int num_of_piles;
    std::vector<int> num_of_objects;
    Player *player;
    Player *agent;

public:

    Model(int num_of_piles, std::vector<int> &num_of_objects, Position position);

    void Set(int num_of_piles, std::vector<int> &num_of_objects, Position position);

    void Run();

};

#endif //NIM_MODEL_H
