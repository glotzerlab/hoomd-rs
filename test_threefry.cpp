#include "../random123/include/Random123/threefry.h"

int main() {
  threefry2x64_ctr_t c = {{0, 0}};
  threefry2x64_key_t k = {{0, 0}};
  threefry2x64_ctr_t result = threefry2x64(c, k);
  std::cout << "[" << result.v[0] << "  " << result.v[1] << "]" << std::endl;
  return 0;
}
