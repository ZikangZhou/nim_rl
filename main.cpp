#include <iostream>

#include "state.h"

int main() {
  State state = {1, 2, 3};
  state = {2, 4};
  std::cout << state << std::endl;
  return 0;
}
