#include "game.h"

int main() {
  Game game(State{10, 20, 10}, new HumanAgent, new OptimalAgent);
  game.Run();
  return 0;
}
