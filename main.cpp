#include "game.h"

int main() {
  Game game(State{10, 10, 10}, new QLearningAgent, new RandomAgent);
  game.Train(200000);
  game.Play();
  return 0;
}
