//
// Created by 周梓康 on 2019/12/26.
//

#include "model.h"

Model::Model(int num_of_piles, std::vector<int> &num_of_objects, Position position) {
    this->num_of_piles = num_of_piles;
    this->num_of_objects = num_of_objects;
    this->player = new Player(position);
    position == OFFENSIVE ? this->agent = new Player(DEFENSIVE) : this->player = new Player(OFFENSIVE);
}

void Model::Set(int num_of_piles, std::vector<int> &num_of_objects, Position position) {
    this->num_of_piles = num_of_piles;
    this->num_of_objects = num_of_objects;
    this->player = new Player(position);
    position == OFFENSIVE ? this->agent = new Player(DEFENSIVE) : this->player = new Player(OFFENSIVE);
}

void Model::Run() {
    if (agent->getPosition() == OFFENSIVE) {
        agent->Action(*this, 0, 0);
    }
}