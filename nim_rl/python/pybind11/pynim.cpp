//
// Created by 周梓康 on 2020/4/10.
//

#include "nim_rl/game.h"
#include "nim_rl/agent.h"
#include "nim_rl/state.h"
#include "nim_rl/action.h"
#include "pybind11/include/pybind11/operators.h"
#include "pybind11/include/pybind11/pybind11.h"

namespace nim_rl {
namespace {

using ::nim_rl::Game;
using ::nim_rl::Agent;
using ::nim_rl::State;
using ::nim_rl::Action;

namespace py = ::pybind11;

PYBIND11_MODULE(pynim, m) {
  m.doc() = "Nim RL";

  m.attr("CHECK_POINT") = py::int_(nim_rl::kCheckPoint);
  m.attr("WIN_REWARD") = py::float_(nim_rl::kWinReward);
  m.attr("TIE_REWARD") = py::float_(nim_rl::kTieReward);
  m.attr("LOSE_REWARD") = py::float_(nim_rl::kLoseReward);
  m.attr("MAX_VALUE") = py::float_(nim_rl::kMaxValue);
  m.attr("MIN_VALUE") = py::float_(nim_rl::kMinValue);
  m.attr("PRECISON") = py::int_(nim_rl::kPrecision);

  py::class_<Game>(m, "Game").def(py::init<>())
      .def(py::init<State>())
      .def(py::init<State, Agent *, Agent *>())
      .def("get_first_player", py::overload_cast<>(&Game::GetFirstPlayer))
      .def("get_first_player",
           py::overload_cast<>(&Game::GetFirstPlayer, py::const_))
      .def("get_initial_state", &Game::GetInitialState)
      .def("get_reward", &Game::GetReward)
      .def("get_second_player", py::overload_cast<>(&Game::GetSecondPlayer))
      .def("get_second_player",
           py::overload_cast<>(&Game::GetSecondPlayer, py::const_))
      .def("get_state", &Game::GetState)
      .def("is_terminal", &Game::IsTerminal)
      .def("play", &Game::Play, py::arg("episodes") = 1)
      .def("render", &Game::Render)
      .def("reset", &Game::Reset)
      .def("set_first_player", &Game::SetFirstPlayer)
      .def("set_initial_state",
           (void (Game::*)(const State &)) &Game::SetInitialState)
      .def("set_reward", &Game::SetReward)
      .def("set_second_player", &Game::SetSecondPlayer)
      .def("set_state", &Game::SetState<const State &>)
      .def("step", &Game::Step)
      .def("train", &Game::Train, py::arg("episodes") = 0);

  m.def("swap", py::overload_cast<Game &, Game &>(&swap));
}
}  // namespace
}  // namespace nim_rl
