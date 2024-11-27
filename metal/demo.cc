#include <iostream>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

int main() {
  const int SIZE = 32;
  const char kernelSrc[] = R"(
    #include <metal_stdlib>
    using namespace metal;

    kernel void Add(
        device float * a [[buffer(0)]],
        device float * b [[buffer(1)]],
        device float * c [[buffer(2)]],
        uint id [[thread_position_in_grid]])
    {
        c[id] = a[id] + b[id];
    }
  )";
  NS::Error * error = nullptr;

  auto device = MTL::CreateSystemDefaultDevice();
  auto commandQueue = device->newCommandQueue();
  auto library =
      device->newLibrary(NS::MakeConstantString(kernelSrc), nullptr, &error);
  auto kernelFunction = library->newFunction(NS::MakeConstantString("Add"));
  auto computePipelineState =
      device->newComputePipelineState(kernelFunction, &error);

  float * a = new float[SIZE];
  float * b = new float[SIZE];
  float * c = new float[SIZE];

  for (int i = 0; i < SIZE; ++i) {
    a[i] = i;
    b[i] = SIZE - i;
  }

  MTL::Buffer * bufferA =
      device->newBuffer(sizeof(float) * SIZE, MTL::ResourceStorageModeShared);
  MTL::Buffer * bufferB =
      device->newBuffer(sizeof(float) * SIZE, MTL::ResourceStorageModeShared);
  MTL::Buffer * bufferC =
      device->newBuffer(sizeof(float) * SIZE, MTL::ResourceStorageModeShared);

  std::memcpy(bufferA->contents(), a, sizeof(float) * SIZE);
  std::memcpy(bufferB->contents(), b, sizeof(float) * SIZE);

  auto commandBuffer = commandQueue->commandBuffer();
  auto computeEncoder = commandBuffer->computeCommandEncoder();
  computeEncoder->setComputePipelineState(computePipelineState);
  computeEncoder->setBuffer(bufferA, 0, 0);
  computeEncoder->setBuffer(bufferB, 0, 1);
  computeEncoder->setBuffer(bufferC, 0, 2);

  MTL::Size gridSize = MTL::Size(SIZE, 1, 1);
  MTL::Size threadGroupSize =
      MTL::Size(computePipelineState->maxTotalThreadsPerThreadgroup(), 1, 1);
  computeEncoder->dispatchThreads(gridSize, threadGroupSize);
  computeEncoder->endEncoding();

  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  std::memcpy(c, bufferC->contents(), sizeof(float) * SIZE);

  std::cout << "Sum: ";
  for (int i = 0; i < SIZE; ++i) { std::cout << c[i] << " "; }
  std::cout << std::endl;

  delete[] a;
  delete[] b;
  delete[] c;
  bufferA->release();
  bufferB->release();
  bufferC->release();
  commandQueue->release();
  device->release();

  return 0;
}
