// #define THREEFRY2x64_DEFAULT_ROUNDS 13
#define THREEFRY4x64_DEFAULT_ROUNDS 12

#include "../random123/include/Random123/threefry.h"

// const long NUM_OUTPUTS = 110423593;
const long NUM_OUTPUTS = 17;

int main() {
  threefry4x64_key_t key = {{0, 0, 0, 0}};
  threefry4x64_ctr_t ctr = {{0, 0, 0, 0}};

  threefry4x64_ctr_t out = threefry4x64(ctr, key);
  // std::cout << "[ " << out.v[0] << " " << out.v[1] << " " << out.v[2] << " "
  //           << out.v[3] << " ]" << std::endl;
  std::cout << "[ " << std::endl;
  uint64_t results[NUM_OUTPUTS];
  long idx = 0;
  while (idx < NUM_OUTPUTS) {
    threefry4x64_ctr_t out = threefry4x64(ctr, key);
    results[idx++] = out.v[0];
    results[idx++] = out.v[1];
    ctr = out;
    std::cout << out.v[0] << ", " << out.v[1] <<", "<< out.v[2] << ", "<<out.v[3]<<",\n";
  }
  std::cout << " ]" << std::endl;
  return 0;
}

// Threefry 2x64
// int main() {
//   std::cout << THREEFRY2x64_DEFAULT_ROUNDS << std::endl;
//   threefry2x64_key_t key = {{0, 0}};
//   threefry2x64_ctr_t ctr = {{0, 0}};

//   // uint64_t results[NUM_OUTPUTS];
//   long idx = 0;

//   while (idx < NUM_OUTPUTS) {
//     if (idx % 100000000 == 0)
//       printf("progress%li\n", idx);

//     threefry2x64_ctr_t out = threefry2x64(ctr, key);
//     // results[idx++] = out.v[0];
//     // if (idx < NUM_OUTPUTS)
//     // results[idx++] = out.v[1];
//     idx++;
//     ctr = out;
//     if (idx == NUM_OUTPUTS - 1)
//       printf("%llu,\n", out.v[0]);
//   }
//   return 0;
// }
