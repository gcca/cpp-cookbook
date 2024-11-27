#include <iostream>

#include <arm_neon.h>

const int SIZE = 32;

void vectorAdd(float * a, float * b, float * c, int size) {
  float32x4_t va, vb, vc;
  int i = 0;

  for (; i < size - (size % 4); i += 4) {
    va = vld1q_f32(&a[i]);
    vb = vld1q_f32(&b[i]);

    vc = vaddq_f32(va, vb);

    vst1q_f32(&c[i], vc);
  }

  for (; i < size; ++i) { c[i] = a[i] + b[i]; }
}

int main() {
  float a[SIZE], b[SIZE], c[SIZE];

  for (int i = 0; i < SIZE; ++i) {
    a[i] = i;
    b[i] = SIZE - i;
  }

  vectorAdd(a, b, c, SIZE);

  std::cout << "Sum: ";
  for (int i = 0; i < SIZE; ++i) { std::cout << c[i] << " "; }
  std::cout << std::endl;

  return 0;
}
