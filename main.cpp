#include "game.h"

int main() {
  State initial_state{10, 10, 10};
  double alpha = 0.5, gamma = 1.0, epsilon = 1.0, decay_factor = 0.9;
  int train_episodes = 50000, play_episodes = 10000;

  Game game(initial_state);
  OptimalAgent optimal_agent;
  HumanAgent human_agent;
  RandomAgent random_agent;
  ValueIterationAgent value_iteration_agent;
  ESMonteCarloAgent es_mc_agent1(gamma);
  ESMonteCarloAgent es_mc_agent2(gamma);
  OnPolicyMonteCarloAgent on_policy_mc_agent1(gamma, 0.1, 1.0);
  OnPolicyMonteCarloAgent on_policy_mc_agent2(gamma, 0.1, 1.0);
  OffPolicyMonteCarloAgent off_policy_mc_agent1(gamma, 0.1, 1.0);
  OffPolicyMonteCarloAgent off_policy_mc_agent2(gamma, 0.1, 1.0);
  QLearningAgent ql_agent1(alpha, gamma, epsilon, decay_factor);
  QLearningAgent ql_agent2(alpha, gamma, epsilon, decay_factor);
  SarsaAgent sarsa_agent1(alpha, gamma, epsilon, decay_factor);
  SarsaAgent sarsa_agent2(alpha, gamma, epsilon, decay_factor);
  ExpectedSarsaAgent expected_sarsa_agent1(alpha, gamma, epsilon, decay_factor);
  ExpectedSarsaAgent expected_sarsa_agent2(alpha, gamma, epsilon, decay_factor);
  DoubleQLearningAgent double_ql_agent1(alpha, gamma, epsilon, decay_factor);
  DoubleQLearningAgent double_ql_agent2(alpha, gamma, epsilon, decay_factor);
  DoubleSarsaAgent double_sarsa_agent1(alpha, gamma, epsilon, decay_factor);
  DoubleSarsaAgent double_sarsa_agent2(alpha, gamma, epsilon, decay_factor);
  DoubleExpectedSarsaAgent
      double_expected_sarsa_agent1(alpha, gamma, epsilon, decay_factor);
  DoubleExpectedSarsaAgent
      double_expected_sarsa_agent2(alpha, gamma, epsilon, decay_factor);

  game.SetFirstPlayer(&value_iteration_agent);
  game.SetSecondPlayer(&optimal_agent);
  game.Train();
  std::cout << value_iteration_agent.GetValues() << std::endl;
  game.Play(play_episodes);

  game.SetFirstPlayer(&es_mc_agent1);
  game.SetSecondPlayer(&optimal_agent);
  game.Train(train_episodes);
  std::cout << es_mc_agent1.GetValues() << std::endl;
  game.SetSecondPlayer(&optimal_agent);
  game.Play(play_episodes);

  game.SetFirstPlayer(&on_policy_mc_agent1);
  game.SetSecondPlayer(&on_policy_mc_agent2);
  game.Train(train_episodes);
  std::cout << on_policy_mc_agent1.GetValues() << std::endl;
  game.SetSecondPlayer(&optimal_agent);
  game.Play(play_episodes);

  game.SetFirstPlayer(&off_policy_mc_agent1);
  game.SetSecondPlayer(&off_policy_mc_agent2);
  game.Train(train_episodes);
  std::cout << off_policy_mc_agent1.GetValues() << std::endl;
  game.SetSecondPlayer(&optimal_agent);
  game.Play(play_episodes);

  game.SetFirstPlayer(&ql_agent1);
  game.SetSecondPlayer(&ql_agent2);
  game.Train(train_episodes);
  std::cout << ql_agent1.GetValues() << std::endl;
  game.SetSecondPlayer(&optimal_agent);
  game.Play(play_episodes);

  game.SetFirstPlayer(&sarsa_agent1);
  game.SetSecondPlayer(&sarsa_agent2);
  game.Train(train_episodes);
  std::cout << sarsa_agent1.GetValues() << std::endl;
  game.SetSecondPlayer(&optimal_agent);
  game.Play(play_episodes);

  game.SetFirstPlayer(&expected_sarsa_agent1);
  game.SetSecondPlayer(&expected_sarsa_agent2);
  game.Train(train_episodes);
  std::cout << expected_sarsa_agent1.GetValues() << std::endl;
  game.SetSecondPlayer(&optimal_agent);
  game.Play(play_episodes);

  game.SetFirstPlayer(&double_ql_agent1);
  game.SetSecondPlayer(&double_ql_agent2);
  game.Train(train_episodes);
  std::cout << double_ql_agent1.GetValues() << std::endl;
  game.SetSecondPlayer(&optimal_agent);
  game.Play(play_episodes);

  game.SetFirstPlayer(&double_sarsa_agent1);
  game.SetSecondPlayer(&double_sarsa_agent2);
  game.Train(train_episodes);
  std::cout << double_sarsa_agent1.GetValues() << std::endl;
  game.SetSecondPlayer(&optimal_agent);
  game.Play(play_episodes);

  game.SetFirstPlayer(&double_expected_sarsa_agent1);
  game.SetSecondPlayer(&double_expected_sarsa_agent2);
  game.Train(train_episodes);
  std::cout << double_expected_sarsa_agent1.GetValues() << std::endl;
  game.SetSecondPlayer(&optimal_agent);
  game.Play(play_episodes);
  return 0;
}
