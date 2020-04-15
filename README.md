# NimRL: A reinforcement learning framework for the game of Nim

NimRL provides the environment for playing [the game of Nim](https://en.wikipedia.org/wiki/Nim)
and a collection of reinforcement learning algorithms. Currently, NimRL supports
most of the algorithms in Chapter 4~7 of
[Reinforcement Learning: An Introduction (2nd Edition)](http://incompleteideas.net/book/RLbook2018.pdf).
Furthermore, any other 2-player games (e.g., board games like Go, Chess, 
and Tic-Tac-Toe) can be easily extended on NimRL. The games and the 
Tabular-based Reinfrocement Learning methods are written in C++, while some 
Deep Reinfrocement Learning methods are written in Python. All of the core APIs 
are exposed to Python using [pybind11](https://github.com/pybind/pybind11).

[Install NimRL](docs/install.md) and start your journal!
