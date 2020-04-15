// Copyright 2020 Zhou Zikang. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "nim_rl/action/action.h"
#include "nim_rl/agent/agent.h"
#include "nim_rl/agent/dp_agent.h"
#include "nim_rl/agent/human_agent.h"
#include "nim_rl/agent/monte_carlo_agent.h"
#include "nim_rl/agent/n_step_bootstrapping_agent.h"
#include "nim_rl/agent/optimal_agent.h"
#include "nim_rl/agent/random_agent.h"
#include "nim_rl/agent/rl_agent.h"
#include "nim_rl/agent/td_agent.h"
#include "nim_rl/environment/game.h"
#include "nim_rl/exploration/exploration.h"
#include "nim_rl/state/state.h"
#include "pybind11/include/pybind11/operators.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"

namespace nim_rl {
namespace {

using Reward = RLAgent::Reward;
using Transitions = std::unordered_map<RLAgent::StateAction,
                                       std::vector<RLAgent::StateProb>>;
using Values = std::unordered_map<State, Reward>;

namespace py = ::pybind11;

template<class ExplorationBase = Exploration>
class PyExploration : public ExplorationBase {
  using ExplorationBase::ExplorationBase;
  Action Explore(const std::vector<nim_rl::Action> &legal_actions,
                 const std::vector<nim_rl::Action> &greedy_actions) override {
    PYBIND11_OVERLOAD_PURE_NAME(Action, ExplorationBase, "explore", Explore,
                                legal_actions, greedy_actions);
  }
  void Update() override {
    PYBIND11_OVERLOAD_PURE_NAME(void, ExplorationBase, "update", Update,);
  }
};

template<class AgentBase = Agent>
class PyAgent : public AgentBase {
 public:
  using Reward = typename AgentBase::Reward;
  using AgentBase::AgentBase;
  ~PyAgent() override = default;
  void Initialize(const std::vector<State> &all_states) override {
    PYBIND11_OVERLOAD_NAME(void, AgentBase, "initialize", Initialize,
                           all_states);
  }
  void Reset() override {
    PYBIND11_OVERLOAD_NAME(void, AgentBase, "reset", Reset,);
  }
  Action Step(Game *game, bool is_evaluation) override {
    PYBIND11_OVERLOAD_NAME(Action, AgentBase, "step", Step, game,
                           is_evaluation);
  }

 protected:
  Action Policy(const State &state, bool is_evaluation) override {
    PYBIND11_OVERLOAD_PURE_NAME(Action, AgentBase, "policy", Policy, state,
                                is_evaluation);
  }
  void Update(const State &update_state, const State &current_state,
              Reward reward) override {
    PYBIND11_OVERLOAD_NAME(void, AgentBase, "update", Update, update_state,
                           current_state, reward);
  }
};

template<class RLAgentBase = RLAgent>
class PyRLAgent : public PyAgent<RLAgentBase> {
 public:
  using PyAgent<RLAgentBase>::PyAgent;
  ~PyRLAgent() override = default;
  std::unordered_map<State, Reward> GetValues() const override {
    PYBIND11_OVERLOAD_NAME(Values, RLAgentBase, "get_values", GetValues,);
  }
  void SetValues(const std::unordered_map<State, Reward> &values) override {
    PYBIND11_OVERLOAD_NAME(void, RLAgentBase, "set_values", SetValues, values);
  }
  void UpdateExploration() override {
    PYBIND11_OVERLOAD_NAME(void, RLAgentBase, "update_exploration",
                           UpdateExploration,);
  }

 protected:
  Action PolicyImpl(const std::vector<Action> &legal_actions,
                    const std::vector<Action> &greedy_actions) override {
    PYBIND11_OVERLOAD_PURE_NAME(Action, RLAgentBase, "policy_impl", PolicyImpl,
                                legal_actions, greedy_actions);
  }
};

template<class MonteCarloAgentBase = MonteCarloAgent>
class PyMonteCarloAgent : public PyRLAgent<MonteCarloAgentBase> {
 public:
  using PyRLAgent<MonteCarloAgentBase>::PyRLAgent;
  ~PyMonteCarloAgent() override = default;
};

template<class TDAgentBase = TDAgent>
class PyTDAgent : public PyRLAgent<TDAgentBase> {
 public:
  using PyRLAgent<TDAgentBase>::PyRLAgent;
  ~PyTDAgent() override = default;
};

template<class DoubleLearningAgentBase = DoubleLearningAgent>
class PyDoubleLearningAgent : public PyTDAgent<DoubleLearningAgentBase> {
 public:
  using PyTDAgent<DoubleLearningAgentBase>::PyTDAgent;
  ~PyDoubleLearningAgent() override = default;

 protected:
  void DoUpdate(const State &update_state, const State &current_state,
                Reward reward,
                std::unordered_map<State, Reward> *values) override {
    PYBIND11_OVERLOAD_PURE_NAME(void, DoubleLearningAgentBase, "do_update",
                                DoUpdate, update_state, current_state, reward,
                                values);
  }
};

template<class NStepBootstrappingAgentBase = NStepBootstrappingAgent>
class PyNStepBootstrappingAgent
    : public PyTDAgent<NStepBootstrappingAgentBase> {
 public:
  using PyTDAgent<NStepBootstrappingAgentBase>::PyTDAgent;
  ~PyNStepBootstrappingAgent() override = default;
};

PYBIND11_MODULE(pynim, m) {
  m.doc() = "NimRL";

  py::class_<Action>(m, "Action").def(py::init<>())
      .def(py::init<int, int>())
      .def("get_num_objects", &Action::GetNumObjects)
      .def("get_pile_id", &Action::GetPileId)
      .def(hash(py::self))
      .def("is_legal", &Action::IsLegal)
      .def("set_num_objects", &Action::SetNumObjects)
      .def("set_pile_id", &Action::SetPileId)
      .def("__eq__",
           py::overload_cast<const Action &, const Action &>(&operator==),
           py::is_operator())
      .def("__ne__",
           py::overload_cast<const Action &, const Action &>(&operator!=),
           py::is_operator())
      .def("__str__", &Action::ToString);

  py::class_<State>(m, "State").def(py::init<>())
      .def(py::init<State::size_type, unsigned>())
      .def(py::init<std::vector<unsigned>>())
      .def("apply_action", &State::ApplyAction)
      .def("child", &State::Child)
      .def("children", &State::Children)
      .def("clear", &State::Clear)
      .def("get_all_states", &State::GetAllStates)
      .def(hash(py::self))
      .def("is_empty", &State::IsEmpty)
      .def("is_terminal", &State::IsTerminal)
      .def("legal_actions", &State::LegalActions)
      .def("nim_sum", &State::NimSum)
      .def("out_of_range", &State::OutOfRange)
      .def("parent", &State::Parent)
      .def("__len__", &State::Size)
      .def("undo_action", &State::UndoAction)
      .def("__eq__",
           py::overload_cast<const State &, const State &>(&operator==),
           py::is_operator())
      .def("__getitem__",
           [](const State &state, std::size_t i) { return state[i]; })
      .def("__ne__",
           py::overload_cast<const State &, const State &>(&operator!=),
           py::is_operator())
      .def("__str__", &State::ToString);

  m.def("swap", py::overload_cast<State &, State &>(&swap));

  m.attr("CHECK_POINT") = py::int_(nim_rl::kCheckPoint);
  m.attr("WIN_REWARD") = py::float_(nim_rl::kWinReward);
  m.attr("TIE_REWARD") = py::float_(nim_rl::kTieReward);
  m.attr("LOSE_REWARD") = py::float_(nim_rl::kLoseReward);
  m.attr("MAX_VALUE") = py::float_(nim_rl::kMaxValue);
  m.attr("MIN_VALUE") = py::float_(nim_rl::kMinValue);
  m.attr("PRECISION") = py::int_(nim_rl::kPrecision);

  py::class_<Game>(m, "Game").def(py::init<>())
      .def(py::init<const State &>())
      .def(py::init<const State &, const Agent &, const Agent &>())
      .def("get_all_states", &Game::GetAllStates)
      .def("get_first_player", &Game::GetFirstPlayer)
      .def("get_initial_state", &Game::GetInitialState)
      .def("get_reward", &Game::GetReward)
      .def("get_second_player", &Game::GetSecondPlayer)
      .def("get_state", &Game::GetState)
      .def("is_terminal", &Game::IsTerminal)
      .def("play", &Game::Play, py::arg("episodes") = 1)
      .def("print_values", &Game::PrintValues)
      .def("render", &Game::Render)
      .def("reset", &Game::Reset)
      .def("set_first_player", &Game::SetFirstPlayer<const Agent &>)
      .def("set_initial_state", &Game::SetInitialState<const State &>)
      .def("set_reward", &Game::SetReward)
      .def("set_second_player", &Game::SetSecondPlayer<const Agent &>)
      .def("set_state", &Game::SetState<const State &>)
      .def("step", &Game::Step)
      .def("train", &Game::Train, py::arg("episodes") = 0);

  m.def("swap", py::overload_cast<Game &, Game &>(&swap));

  py::class_<Exploration, PyExploration<>>(m, "Exploration")
      .def("explore", &Exploration::Explore)
      .def("update", &Exploration::Update);

  py::class_<EpsilonGreedy, Exploration, PyExploration<EpsilonGreedy>>(
      m, "EpsilonGreedy")
      .def(py::init<double, double, double>(),
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("explore", &EpsilonGreedy::Explore)
      .def("get_epsilon", &EpsilonGreedy::GetEpsilon)
      .def("get_epsilon_decay_factor", &EpsilonGreedy::GetEpsilonDecayFactor)
      .def("get_min_epsilon", &EpsilonGreedy::GetMinEpsilon)
      .def("set_epsilon", &EpsilonGreedy::SetEpsilon)
      .def("set_epsilon_decay_factor", &EpsilonGreedy::SetEpsilonDecayFactor)
      .def("set_min_epsilon", &EpsilonGreedy::SetMinEpsilon)
      .def("update", &EpsilonGreedy::Update);

  py::class_<Agent, PyAgent<>>(m, "Agent").def("initialize", &Agent::Initialize)
      .def("reset", &Agent::Reset)
      .def("step", &Agent::Step);

  py::class_<HumanAgent, Agent, PyAgent<HumanAgent>>(m, "HumanAgent")
      .def(py::init<>());

  py::class_<OptimalAgent, Agent, PyAgent<OptimalAgent>>(m, "OptimalAgent")
      .def(py::init<>())
      .def("policy", &OptimalAgent::Policy);

  py::class_<RandomAgent, Agent, PyAgent<RandomAgent>>(m, "RandomAgent")
      .def(py::init<>());

  py::class_<RLAgent, Agent, PyRLAgent<>>(m, "RLAgent")
      .def("get_values", &RLAgent::GetValues)
      .def("initialize", &RLAgent::Initialize)
      .def("optimal_action_ratios", &RLAgent::OptimalActionsRatio)
      .def("reset", &RLAgent::Reset)
      .def("set_values",
           py::overload_cast<const Values &>(&RLAgent::SetValues));

  py::class_<DPAgent, RLAgent, PyRLAgent<DPAgent>>(m, "DPAgent")
      .def(py::init<double, double>(),
           py::arg("gamma") = kDefaultGamma,
           py::arg("threshold") = kDefaultThreshold)
      .def("get_gamma", &DPAgent::GetGamma)
      .def("get_threshold", &DPAgent::GetThreshold)
      .def("get_transitions", &DPAgent::GetTransitions)
      .def("initialize", &DPAgent::Initialize)
      .def("set_gamma", &DPAgent::SetGamma)
      .def("set_threshold", &DPAgent::SetThreshold)
      .def("set_transitions",
           &DPAgent::SetTransitions<const Transitions &>);

  py::class_<PolicyIterationAgent, DPAgent, PyRLAgent<PolicyIterationAgent>>(
      m, "PolicyIterationAgent")
      .def(py::init<double, double>(),
           py::arg("gamma") = kDefaultGamma,
           py::arg("threshold") = kDefaultThreshold)
      .def("initialize", &PolicyIterationAgent::Initialize);

  py::class_<ValueIterationAgent, DPAgent, PyRLAgent<ValueIterationAgent>>(
      m, "ValueIterationAgent")
      .def(py::init<double, double>(),
           py::arg("gamma") = kDefaultGamma,
           py::arg("threshold") = kDefaultThreshold)
      .def("initialize", &ValueIterationAgent::Initialize);

  py::enum_<ImportanceSampling>(m, "ImportanceSampling")
      .value("WEIGHTED", ImportanceSampling::kWeighted)
      .value("NORMAL", ImportanceSampling::kNormal);

  py::class_<MonteCarloAgent, RLAgent, PyMonteCarloAgent<>>(m,
                                                            "MonteCarloAgent")
      .def(py::init<double>(), py::arg("gamma") = kDefaultGamma)
      .def("get_gamma", &MonteCarloAgent::GetGamma)
      .def("initialize", &MonteCarloAgent::Initialize)
      .def("reset", &MonteCarloAgent::Reset)
      .def("set_gamma", &MonteCarloAgent::SetGamma)
      .def("step", &MonteCarloAgent::Step);

  py::class_<ESMonteCarloAgent,
             MonteCarloAgent,
             PyMonteCarloAgent<ESMonteCarloAgent>>(m, "ESMonteCarloAgent")
      .def(py::init<double>(), py::arg("gamma") = kDefaultGamma)
      .def("step", &MonteCarloAgent::Step);

  py::class_<OnPolicyMonteCarloAgent,
             MonteCarloAgent,
             PyMonteCarloAgent<OnPolicyMonteCarloAgent>>(
      m, "OnPolicyMonteCarloAgent")
      .def(py::init<double, const Exploration &>(),
           py::arg("gamma") = kDefaultGamma,
           py::arg("exploration") = EpsilonGreedy())
      .def("update_exploration", &OnPolicyMonteCarloAgent::UpdateExploration);

  py::class_<OffPolicyMonteCarloAgent,
             MonteCarloAgent,
             PyMonteCarloAgent<OffPolicyMonteCarloAgent>>(
      m, "OffPolicyMonteCarloAgent")
      .def(py::init<double, ImportanceSampling, double, double, double>(),
           py::arg("gamma") = kDefaultGamma,
           py::arg("importance_sampling") = ImportanceSampling::kWeighted,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("update_exploration", &OffPolicyMonteCarloAgent::UpdateExploration);

  py::class_<TDAgent, RLAgent, PyTDAgent<>>(m, "TDAgent")
      .def(py::init<double, double, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("get_alpha", &TDAgent::GetAlpha)
      .def("get_gamma", &TDAgent::GetGamma)
      .def("set_alpha", &TDAgent::SetAlpha)
      .def("set_gamma", &TDAgent::SetGamma)
      .def("step", &TDAgent::Step)
      .def("update_exploration", &TDAgent::UpdateExploration);

  py::class_<QLearningAgent, TDAgent, PyTDAgent<QLearningAgent>>(
      m, "QLearningAgent")
      .def(py::init<double, double, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon);

  py::class_<SarsaAgent, TDAgent, PyTDAgent<SarsaAgent>>(m, "SarsaAgent")
      .def(py::init<double, double, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon);

  py::class_<ExpectedSarsaAgent, TDAgent, PyTDAgent<ExpectedSarsaAgent>>(
      m, "ExpectedSarsaAgent")
      .def(py::init<double, double, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("reset", &ExpectedSarsaAgent::Reset);

  py::class_<DoubleLearningAgent, TDAgent, PyDoubleLearningAgent<>>(
      m, "DoubleLearningAgent")
      .def("get_values", &DoubleLearningAgent::GetValues)
      .def("initialize", &DoubleLearningAgent::Initialize)
      .def("reset", &DoubleLearningAgent::Reset)
      .def("set_values",
           py::overload_cast<const Values &>(&DoubleLearningAgent::SetValues));

  py::class_<DoubleQLearningAgent,
             DoubleLearningAgent,
             PyDoubleLearningAgent<DoubleQLearningAgent>>(
      m, "DoubleQLearningAgent")
      .def(py::init<double, double, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon);

  py::class_<DoubleSarsaAgent,
             DoubleLearningAgent,
             PyDoubleLearningAgent<DoubleSarsaAgent>>(
      m, "DoubleSarsaAgent")
      .def(py::init<double, double, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon);

  py::class_<DoubleExpectedSarsaAgent,
             DoubleLearningAgent,
             PyDoubleLearningAgent<DoubleExpectedSarsaAgent>>(
      m, "DoubleExpectedSarsaAgent")
      .def(py::init<double, double, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("reset", &DoubleExpectedSarsaAgent::Reset);

  py::class_<NStepBootstrappingAgent, TDAgent, PyNStepBootstrappingAgent<>>(
      m, "NStepBootstrappingAgent")
      .def("get_n", &NStepBootstrappingAgent::GetN)
      .def("reset", &NStepBootstrappingAgent::Reset)
      .def("set_n", &NStepBootstrappingAgent::SetN)
      .def("step", &NStepBootstrappingAgent::Step);

  py::class_<NStepSarsaAgent,
             NStepBootstrappingAgent,
             PyNStepBootstrappingAgent<NStepSarsaAgent>>(
      m, "NStepSarsaAgent")
      .def(py::init<double, double, int, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("n") = kDefaultN,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon);

  py::class_<NStepExpectedSarsaAgent,
             NStepBootstrappingAgent,
             PyNStepBootstrappingAgent<NStepExpectedSarsaAgent>>(
      m, "NStepExpectedSarsaAgent")
      .def(py::init<double, double, int, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("n") = kDefaultN,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("reset", &NStepExpectedSarsaAgent::Reset);

  py::class_<OffPolicyNStepSarsaAgent,
             NStepBootstrappingAgent,
             PyNStepBootstrappingAgent<OffPolicyNStepSarsaAgent>>(
      m, "OffPolicyNStepSarsaAgent")
      .def(py::init<double, double, int, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("n") = kDefaultN,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon);

  py::class_<OffPolicyNStepExpectedSarsaAgent,
             NStepBootstrappingAgent,
             PyNStepBootstrappingAgent<OffPolicyNStepExpectedSarsaAgent>>(
      m, "OffPolicyNStepExpectedSarsaAgent")
      .def(py::init<double, double, int, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("n") = kDefaultN,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("reset", &OffPolicyNStepExpectedSarsaAgent::Reset);

  py::class_<NStepTreeBackupAgent,
             NStepBootstrappingAgent,
             PyNStepBootstrappingAgent<NStepTreeBackupAgent>>(
      m, "NStepTreeBackupAgent")
      .def(py::init<double, double, int, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("n") = kDefaultN,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon);
}

}  // namespace
}  // namespace nim_rl
