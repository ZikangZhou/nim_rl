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

template<typename T>
class SmartPtr {
 public:
  using element_type = T;
  explicit SmartPtr(T *p) {
    PyObject *obj = pybind11::cast(p).ptr();
    Py_INCREF(obj);
    std::shared_ptr<PyObject> ptr(obj, [](PyObject *obj) { Py_DECREF(obj); });
    ptr_ = std::shared_ptr<T>(ptr, p);
  }
  explicit SmartPtr(std::shared_ptr<T> p) : ptr_(p) {}
  explicit operator std::shared_ptr<T>() { return ptr_; }
  T *get() const { return ptr_.get(); }

 private:
  std::shared_ptr<T> ptr_;
};

PYBIND11_DECLARE_HOLDER_TYPE(T, SmartPtr<T>);

namespace nim_rl {
namespace {

using Reward = Agent::Reward;
using Transitions = std::unordered_map<RLAgent::StateAction,
                                       std::vector<RLAgent::StateProb>>;
using Values = RLAgent::Values;

namespace py = ::pybind11;

template<class ExplorationBase = Exploration>
class PyExploration : public ExplorationBase {
 public:
  using ExplorationBase::ExplorationBase;
  explicit PyExploration(const ExplorationBase &exploration_base)
      : ExplorationBase(exploration_base) {}
  ~PyExploration() override = default;
  std::shared_ptr<Exploration> Clone() const override {
    py::object obj = py::cast(this).attr("clone")();
    auto keep_python_state_alive = std::make_shared<py::object>(obj);
    auto ptr = obj.cast<PyExploration *>();
    return std::shared_ptr<Exploration>(keep_python_state_alive, ptr);
  }
  Action Explore(const std::vector<nim_rl::Action> &legal_actions,
                 const std::vector<nim_rl::Action> &greedy_actions) override {
    PYBIND11_OVERLOAD_PURE_NAME(Action, ExplorationBase, "explore", Explore,
                                legal_actions, greedy_actions);
  }
  void Update(int episode) override {
    PYBIND11_OVERLOAD_PURE_NAME(void, ExplorationBase, "update", Update,
                                episode);
  }
};

template<class AgentBase = Agent>
class PyAgent : public AgentBase {
 public:
  using Reward = typename AgentBase::Reward;
  using AgentBase::AgentBase;
  explicit PyAgent(const AgentBase &agent_base) : AgentBase(agent_base) {}
  ~PyAgent() override = default;
  std::shared_ptr<Agent> Clone() const override {
    py::object obj = py::cast(this).attr("clone")();
    auto keep_python_state_alive = std::make_shared<py::object>(obj);
    auto ptr = obj.cast<PyAgent *>();
    return std::shared_ptr<Agent>(keep_python_state_alive, ptr);
  }
  void Initialize(const std::vector<State> &all_states) override {
    PYBIND11_OVERLOAD_NAME(void, AgentBase, "initialize", Initialize,
                           all_states);
  }
  Action Policy(const State &state, bool is_evaluation) override {
    PYBIND11_OVERLOAD_PURE_NAME(Action, AgentBase, "policy", Policy, state,
                                is_evaluation);
  }
  void Reset() override {
    PYBIND11_OVERLOAD_NAME(void, AgentBase, "reset", Reset,);
  }
  Action Step(Game *game, bool is_evaluation) override {
    PYBIND11_OVERLOAD_NAME(Action, AgentBase, "step", Step, game,
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
  explicit PyRLAgent(const PyAgent<RLAgentBase> &rl_agent_base)
      : PyAgent<RLAgentBase>(rl_agent_base) {}
  ~PyRLAgent() override = default;
  std::shared_ptr<Agent> Clone() const override {
    py::object obj = py::cast(this).attr("clone")();
    auto keep_python_state_alive = std::make_shared<py::object>(obj);
    auto ptr = obj.cast<PyRLAgent *>();
    return std::shared_ptr<Agent>(keep_python_state_alive, ptr);
  }
  Values GetValues() const override {
    PYBIND11_OVERLOAD_NAME(Values, RLAgentBase, "get_values", GetValues,);
  }
  Action PolicyImpl(const std::vector<Action> &legal_actions,
                    const std::vector<Action> &greedy_actions) override {
    PYBIND11_OVERLOAD_PURE_NAME(Action, RLAgentBase, "policy_impl", PolicyImpl,
                                legal_actions, greedy_actions);
  }
  void SetValues(const Values &values) override {
    PYBIND11_OVERLOAD_NAME(void, RLAgentBase, "set_values", SetValues, values);
  }
  void UpdateExploration(int episode) override {
    PYBIND11_OVERLOAD_NAME(void, RLAgentBase, "update_exploration",
                           UpdateExploration, episode);
  }
};

template<class DPAgentBase = DPAgent>
class PyDPAgent : public PyRLAgent<DPAgentBase> {
 public:
  using PyRLAgent<DPAgentBase>::PyRLAgent;
  ~PyDPAgent() override = default;
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
  void DoUpdate(const State &update_state, const State &current_state,
                Reward reward, Values *values) override {
    PYBIND11_OVERLOAD_PURE_NAME(void, DoubleLearningAgentBase, "do_update",
                                DoUpdate, update_state, current_state,
                                reward, values);
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

  py::class_<Game, std::shared_ptr<Game>>(m, "Game").def(py::init<>())
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
      .def("set_first_player", &Game::SetFirstPlayer)
      .def("set_initial_state", &Game::SetInitialState<const State &>)
      .def("set_reward", &Game::SetReward)
      .def("set_second_player", &Game::SetSecondPlayer)
      .def("set_state", &Game::SetState<const State &>)
      .def("step", &Game::Step)
      .def("train", &Game::Train, py::arg("episodes") = 0);

  m.def("swap", py::overload_cast<Game &, Game &>(&swap));

  py::class_<Exploration, PyExploration<>, std::shared_ptr<Exploration>>(
      m, "Exploration")
      .def(py::init<>())
      .def(py::init<const Exploration &>(), py::arg("exploration"))
      .def("explore", &Exploration::Explore, py::arg("legal_actions"),
           py::arg("greedy_actions"))
      .def("update", &Exploration::Update);

  py::class_<EpsilonGreedy,
             Exploration,
             PyExploration<EpsilonGreedy>,
             std::shared_ptr<EpsilonGreedy>>(m, "EpsilonGreedy")
      .def(py::init<double, double, double>(),
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("clone", &EpsilonGreedy::Clone)
      .def("explore", &EpsilonGreedy::Explore, py::arg("legal_actions"),
           py::arg("greedy_actions"))
      .def("get_epsilon", &EpsilonGreedy::GetEpsilon)
      .def("get_epsilon_decay_factor", &EpsilonGreedy::GetEpsilonDecayFactor)
      .def("get_min_epsilon", &EpsilonGreedy::GetMinEpsilon)
      .def("set_epsilon", &EpsilonGreedy::SetEpsilon, py::arg("epsilon"))
      .def("set_epsilon_decay_factor", &EpsilonGreedy::SetEpsilonDecayFactor,
           py::arg("epsilon_decay_factor"))
      .def("set_min_epsilon", &EpsilonGreedy::SetMinEpsilon,
           py::arg("min_epsilon"))
      .def("update", &EpsilonGreedy::Update);

  m.def("sample_action", &SampleAction, py::arg("actions"));
  m.def("sample_state", &SampleState, py::arg("states"));

  py::class_<Agent, PyAgent<>, SmartPtr<Agent>>(m, "Agent")
      .def(py::init<>())
      .def(py::init<const Agent &>(), py::arg("agent"))
      .def("get_current_state", &Agent::GetCurrentState)
      .def("initialize", &Agent::Initialize, py::arg("all_states"))
      .def("reset", &Agent::Reset)
      .def("set_current_state", &Agent::SetCurrentState, py::arg("state"))
      .def("step", &Agent::Step, py::arg("game"), py::arg("is_evaluation"))
      .def("update", &Agent::Update, py::arg("update_state"),
           py::arg("current_state"), py::arg("reward"))
      .def_property("_current_state", &Agent::GetCurrentState,
                    &Agent::SetCurrentState);

  py::class_<HumanAgent,
             Agent,
             PyAgent<HumanAgent>,
             SmartPtr<HumanAgent>>(m, "HumanAgent")
      .def(py::init<>())
      .def("clone", &HumanAgent::Clone)
      .def("policy", &HumanAgent::Policy, py::arg("state"),
           py::arg("is_evaluation"));

  py::class_<OptimalAgent,
             Agent,
             PyAgent<OptimalAgent>,
             SmartPtr<OptimalAgent>>(m, "OptimalAgent")
      .def(py::init<>())
      .def("clone", &OptimalAgent::Clone)
      .def("policy", &OptimalAgent::Policy, py::arg("state"),
           py::arg("is_evaluation"));

  py::class_<RandomAgent,
             Agent,
             PyAgent<RandomAgent>,
             SmartPtr<RandomAgent>>(m, "RandomAgent")
      .def(py::init<>())
      .def("clone", &RandomAgent::Clone)
      .def("policy", &RandomAgent::Policy, py::arg("state"),
           py::arg("is_evaluation"));

  py::class_<RLAgent, Agent, PyRLAgent<>, SmartPtr<RLAgent>>(m, "RLAgent")
      .def(py::init<>())
      .def(py::init<const RLAgent &>(), py::arg("agent"))
      .def("add_greedy_action", &RLAgent::AddGreedyAction, py::arg("action"))
      .def("clear_greedy_actions", &RLAgent::ClearGreedyActions)
      .def("get_greedy_actions", &RLAgent::GetGreedyActions)
      .def("get_greedy_value", &RLAgent::GetGreedyValue)
      .def("get_legal_actions", &RLAgent::GetLegalActions)
      .def("get_values", &RLAgent::GetValues)
      .def("initialize", &RLAgent::Initialize, py::arg("all_states"))
      .def("optimal_action_ratios", &RLAgent::OptimalActionsRatio)
      .def("policy", &RLAgent::Policy, py::arg("state"),
           py::arg("is_evaluation"))
      .def("policy_impl", &RLAgent::PolicyImpl, py::arg("legal_actions"),
           py::arg("greedy_actions"))
      .def("reset", &RLAgent::Reset)
      .def("set_greedy_actions", &RLAgent::SetGreedyActions,
           py::arg("greedy_actions"))
      .def("set_greedy_value", &RLAgent::SetGreedyValue,
           py::arg("greedy_value"))
      .def("set_legal_actions", &RLAgent::SetLegalActions,
           py::arg("legal_actions"))
      .def("set_values", &RLAgent::SetValues, py::arg("values"))
      .def("update_exploration", &RLAgent::UpdateExploration,
           py::arg("episode"))
      .def_property("_greedy_value", &RLAgent::GetGreedyValue,
                    &RLAgent::SetGreedyValue)
      .def_property("_legal_actions", &RLAgent::GetLegalActions,
                    &RLAgent::SetLegalActions)
      .def_property("_greedy_actions", &RLAgent::GetGreedyActions,
                    &RLAgent::SetGreedyActions);

  py::class_<DPAgent,
             RLAgent,
             PyDPAgent<>,
             SmartPtr<DPAgent>>(m, "DPAgent")
      .def(py::init<double, double>(),
           py::arg("gamma") = kDefaultGamma,
           py::arg("threshold") = kDefaultThreshold)
      .def("clone", &DPAgent::Clone)
      .def("get_gamma", &DPAgent::GetGamma)
      .def("get_threshold", &DPAgent::GetThreshold)
      .def("get_transitions", &DPAgent::GetTransitions)
      .def("initialize", &DPAgent::Initialize, py::arg("all_states"))
      .def("policy_impl", &DPAgent::PolicyImpl, py::arg("legal_actions"),
           py::arg("greedy_actions"))
      .def("set_gamma", &DPAgent::SetGamma, py::arg("gamma"))
      .def("set_threshold", &DPAgent::SetThreshold, py::arg("threshold"))
      .def("set_transitions", &DPAgent::SetTransitions<const Transitions &>,
           py::arg("transitions"));

  py::class_<PolicyIterationAgent,
             DPAgent,
             PyRLAgent<PolicyIterationAgent>,
             SmartPtr<PolicyIterationAgent>>(m, "PolicyIterationAgent")
      .def(py::init<double, double>(),
           py::arg("gamma") = kDefaultGamma,
           py::arg("threshold") = kDefaultThreshold)
      .def("clone", &PolicyIterationAgent::Clone)
      .def("policy", &PolicyIterationAgent::Policy, py::arg("state"),
           py::arg("is_evaluation"))
      .def("initialize", &PolicyIterationAgent::Initialize,
           py::arg("all_states"));

  py::class_<ValueIterationAgent,
             DPAgent,
             PyRLAgent<ValueIterationAgent>,
             SmartPtr<ValueIterationAgent>>(m, "ValueIterationAgent")
      .def(py::init<double, double>(),
           py::arg("gamma") = kDefaultGamma,
           py::arg("threshold") = kDefaultThreshold)
      .def("clone", &ValueIterationAgent::Clone)
      .def("initialize", &ValueIterationAgent::Initialize,
           py::arg("all_states"));

  py::enum_<ImportanceSampling>(m, "ImportanceSampling")
      .value("WEIGHTED", ImportanceSampling::kWeighted)
      .value("NORMAL", ImportanceSampling::kNormal);

  py::class_<MonteCarloAgent,
             RLAgent,
             PyMonteCarloAgent<>,
             SmartPtr<MonteCarloAgent>>(m, "MonteCarloAgent")
      .def(py::init<double>(), py::arg("gamma") = kDefaultGamma)
      .def("clone", &MonteCarloAgent::Clone)
      .def("get_gamma", &MonteCarloAgent::GetGamma)
      .def("initialize", &MonteCarloAgent::Initialize, py::arg("all_states"))
      .def("policy_impl", &MonteCarloAgent::PolicyImpl,
           py::arg("legal_actions"), py::arg("greedy_actions"))
      .def("reset", &MonteCarloAgent::Reset)
      .def("set_gamma", &MonteCarloAgent::SetGamma, py::arg("gamma"))
      .def("step", &MonteCarloAgent::Step, py::arg("game"),
           py::arg("is_evaluation"));

  py::class_<ESMonteCarloAgent,
             MonteCarloAgent,
             PyMonteCarloAgent<ESMonteCarloAgent>,
             SmartPtr<ESMonteCarloAgent>>(m, "ESMonteCarloAgent")
      .def(py::init<double>(), py::arg("gamma") = kDefaultGamma)
      .def("clone", &ESMonteCarloAgent::Clone)
      .def("step", &ESMonteCarloAgent::Step, py::arg("game"),
           py::arg("is_evaluation"));

  py::class_<OnPolicyMonteCarloAgent,
             MonteCarloAgent,
             PyMonteCarloAgent<OnPolicyMonteCarloAgent>,
             SmartPtr<OnPolicyMonteCarloAgent>>(m, "OnPolicyMonteCarloAgent")
      .def(py::init<double, const Exploration &>(),
           py::arg("gamma") = kDefaultGamma,
           py::arg("exploration") = EpsilonGreedy())
      .def("clone", &OnPolicyMonteCarloAgent::Clone)
      .def("policy_impl", &OnPolicyMonteCarloAgent::PolicyImpl,
           py::arg("legal_actions"), py::arg("greedy_actions"))
      .def("update_exploration", &OnPolicyMonteCarloAgent::UpdateExploration,
           py::arg("episode"));

  py::class_<OffPolicyMonteCarloAgent,
             MonteCarloAgent,
             PyMonteCarloAgent<OffPolicyMonteCarloAgent>,
             SmartPtr<OffPolicyMonteCarloAgent>>(m, "OffPolicyMonteCarloAgent")
      .def(py::init<double, ImportanceSampling, double, double, double>(),
           py::arg("gamma") = kDefaultGamma,
           py::arg("importance_sampling") = ImportanceSampling::kWeighted,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("clone", &OffPolicyMonteCarloAgent::Clone)
      .def("policy_impl", &OffPolicyMonteCarloAgent::PolicyImpl,
           py::arg("legal_actions"), py::arg("greedy_actions"))
      .def("update", &OffPolicyMonteCarloAgent::Update, py::arg("update_state"),
           py::arg("current_state"), py::arg("reward"))
      .def("update_exploration", &OffPolicyMonteCarloAgent::UpdateExploration,
           py::arg("episode"));

  py::class_<TDAgent, RLAgent, PyTDAgent<>, SmartPtr<TDAgent>>(m, "TDAgent")
      .def(py::init<double, double, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("clone", &TDAgent::Clone)
      .def("get_alpha", &TDAgent::GetAlpha)
      .def("get_gamma", &TDAgent::GetGamma)
      .def("policy_impl", &TDAgent::PolicyImpl, py::arg("legal_actions"),
           py::arg("greedy_actions"))
      .def("set_alpha", &TDAgent::SetAlpha, py::arg("alpha"))
      .def("set_gamma", &TDAgent::SetGamma, py::arg("gamma"))
      .def("step", &TDAgent::Step, py::arg("game"), py::arg("is_evaluation"))
      .def("update_exploration", &TDAgent::UpdateExploration,
           py::arg("episode"));

  py::class_<QLearningAgent,
             TDAgent,
             PyTDAgent<QLearningAgent>,
             SmartPtr<QLearningAgent>>(m, "QLearningAgent")
      .def(py::init<double, double, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("clone", &QLearningAgent::Clone)
      .def("update", &QLearningAgent::Update, py::arg("update_state"),
           py::arg("current_state"), py::arg("reward"));

  py::class_<SarsaAgent,
             TDAgent,
             PyTDAgent<SarsaAgent>,
             SmartPtr<SarsaAgent>>(m, "SarsaAgent")
      .def(py::init<double, double, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("clone", &SarsaAgent::Clone)
      .def("update", &SarsaAgent::Update, py::arg("update_state"),
           py::arg("current_state"), py::arg("reward"));

  py::class_<ExpectedSarsaAgent,
             TDAgent,
             PyTDAgent<ExpectedSarsaAgent>,
             SmartPtr<ExpectedSarsaAgent>>(m, "ExpectedSarsaAgent")
      .def(py::init<double, double, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("clone", &ExpectedSarsaAgent::Clone)
      .def("policy", &ExpectedSarsaAgent::Policy, py::arg("state"),
           py::arg("is_evaluation"))
      .def("reset", &ExpectedSarsaAgent::Reset)
      .def("update", &ExpectedSarsaAgent::Update, py::arg("update_state"),
           py::arg("current_state"), py::arg("reward"));

  py::class_<DoubleLearningAgent,
             TDAgent,
             PyDoubleLearningAgent<>,
             SmartPtr<DoubleLearningAgent>>(m, "DoubleLearningAgent")
      .def(py::init<double, double, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("get_values", &DoubleLearningAgent::GetValues)
      .def("initialize", &DoubleLearningAgent::Initialize,
           py::arg("all_states"))
      .def("policy", &DoubleLearningAgent::Policy, py::arg("state"),
           py::arg("is_evaluation"))
      .def("reset", &DoubleLearningAgent::Reset)
      .def("set_values", &DoubleLearningAgent::SetValues, py::arg("values"))
      .def("update", &DoubleLearningAgent::Update, py::arg("update_state"),
           py::arg("current_state"), py::arg("reward"));

  py::class_<DoubleQLearningAgent,
             DoubleLearningAgent,
             PyDoubleLearningAgent<DoubleQLearningAgent>,
             SmartPtr<DoubleQLearningAgent>>(m, "DoubleQLearningAgent")
      .def(py::init<double, double, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("clone", &DoubleQLearningAgent::Clone)
      .def("do_update", &DoubleQLearningAgent::DoUpdate,
           py::arg("update_state"), py::arg("current_state"), py::arg("reward"),
           py::arg("values"));

  py::class_<DoubleSarsaAgent,
             DoubleLearningAgent,
             PyDoubleLearningAgent<DoubleSarsaAgent>,
             SmartPtr<DoubleSarsaAgent>>(m, "DoubleSarsaAgent")
      .def(py::init<double, double, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("clone", &DoubleSarsaAgent::Clone)
      .def("do_update", &DoubleSarsaAgent::DoUpdate, py::arg("update_state"),
           py::arg("current_state"), py::arg("reward"), py::arg("values"));

  py::class_<DoubleExpectedSarsaAgent,
             DoubleLearningAgent,
             PyDoubleLearningAgent<DoubleExpectedSarsaAgent>,
             SmartPtr<DoubleExpectedSarsaAgent>>(m, "DoubleExpectedSarsaAgent")
      .def(py::init<double, double, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("clone", &DoubleExpectedSarsaAgent::Clone)
      .def("do_update", &DoubleExpectedSarsaAgent::DoUpdate,
           py::arg("update_state"), py::arg("current_state"), py::arg("reward"),
           py::arg("values"))
      .def("policy", &DoubleExpectedSarsaAgent::Policy, py::arg("state"),
           py::arg("is_evaluation"))
      .def("reset", &DoubleExpectedSarsaAgent::Reset);

  py::class_<NStepBootstrappingAgent,
             TDAgent,
             PyNStepBootstrappingAgent<>,
             SmartPtr<NStepBootstrappingAgent>>(m, "NStepBootstrappingAgent")
      .def(py::init<double, double, int, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("n") = kDefaultN,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("clone", &NStepBootstrappingAgent::Clone)
      .def("get_n", &NStepBootstrappingAgent::GetN)
      .def("reset", &NStepBootstrappingAgent::Reset)
      .def("set_n", &NStepBootstrappingAgent::SetN, py::arg("n"))
      .def("step", &NStepBootstrappingAgent::Step, py::arg("game"),
           py::arg("is_evaluation"));

  py::class_<NStepSarsaAgent,
             NStepBootstrappingAgent,
             PyNStepBootstrappingAgent<NStepSarsaAgent>,
             SmartPtr<NStepSarsaAgent>>(m, "NStepSarsaAgent")
      .def(py::init<double, double, int, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("n") = kDefaultN,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("clone", &NStepSarsaAgent::Clone)
      .def("update", &NStepSarsaAgent::Update, py::arg("update_state"),
           py::arg("current_state"), py::arg("reward"));

  py::class_<NStepExpectedSarsaAgent,
             NStepBootstrappingAgent,
             PyNStepBootstrappingAgent<NStepExpectedSarsaAgent>,
             SmartPtr<NStepExpectedSarsaAgent>>(m, "NStepExpectedSarsaAgent")
      .def(py::init<double, double, int, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("n") = kDefaultN,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("clone", &NStepExpectedSarsaAgent::Clone)
      .def("policy", &NStepExpectedSarsaAgent::Policy, py::arg("state"),
           py::arg("is_evaluation"))
      .def("reset", &NStepExpectedSarsaAgent::Reset)
      .def("update", &NStepExpectedSarsaAgent::Update, py::arg("update_state"),
           py::arg("current_state"), py::arg("reward"));

  py::class_<OffPolicyNStepSarsaAgent,
             NStepBootstrappingAgent,
             PyNStepBootstrappingAgent<OffPolicyNStepSarsaAgent>,
             SmartPtr<OffPolicyNStepSarsaAgent>>(m, "OffPolicyNStepSarsaAgent")
      .def(py::init<double, double, int, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("n") = kDefaultN,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("clone", &OffPolicyNStepSarsaAgent::Clone)
      .def("update", &OffPolicyNStepSarsaAgent::Update, py::arg("update_state"),
           py::arg("current_state"), py::arg("reward"));

  py::class_<OffPolicyNStepExpectedSarsaAgent,
             NStepBootstrappingAgent,
             PyNStepBootstrappingAgent<OffPolicyNStepExpectedSarsaAgent>,
             SmartPtr<OffPolicyNStepExpectedSarsaAgent>>(
      m, "OffPolicyNStepExpectedSarsaAgent")
      .def(py::init<double, double, int, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("n") = kDefaultN,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("clone", &OffPolicyNStepExpectedSarsaAgent::Clone)
      .def("policy", &OffPolicyNStepExpectedSarsaAgent::Policy,
           py::arg("state"), py::arg("is_evaluation"))
      .def("reset", &OffPolicyNStepExpectedSarsaAgent::Reset)
      .def("update", &OffPolicyNStepExpectedSarsaAgent::Update,
           py::arg("update_state"), py::arg("current_state"),
           py::arg("reward"));

  py::class_<NStepTreeBackupAgent,
             NStepBootstrappingAgent,
             PyNStepBootstrappingAgent<NStepTreeBackupAgent>,
             SmartPtr<NStepTreeBackupAgent>>(m, "NStepTreeBackupAgent")
      .def(py::init<double, double, int, double, double, double>(),
           py::arg("alpha") = kDefaultAlpha,
           py::arg("gamma") = kDefaultGamma,
           py::arg("n") = kDefaultN,
           py::arg("epsilon") = kDefaultEpsilon,
           py::arg("epsilon_decay_factor") = kDefaultEpsilonDecayFactor,
           py::arg("min_epsilon") = kDefaultMinEpsilon)
      .def("clone", &NStepTreeBackupAgent::Clone)
      .def("update", &NStepTreeBackupAgent::Update, py::arg("update_state"),
           py::arg("current_state"), py::arg("reward"));
}

}  // namespace
}  // namespace nim_rl
