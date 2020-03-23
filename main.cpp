#include "game.h"

int main() {
  QLearningAgent q_agent(0.99, 1.0, 1.0, 0.9);
  QLearningAgent q_agent2(0.99, 1.0, 1.0, 0.9);
  OptimalAgent optimal_agent;
  RandomAgent random_agent;
  Game game(State{10, 10, 10}, &q_agent, &q_agent2);
  game.Train(20000);
  std::cout << q_agent.GetQValues() << std::endl;
  q_agent.SetEpsilon(0.0);
  game.SetSecondPlayer(&optimal_agent);
  game.Play(50000);
  return 0;
}
