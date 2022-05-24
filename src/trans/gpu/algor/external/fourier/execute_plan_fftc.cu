#include "cufft.h"
#include "stdio.h"
#include <iostream>
#include <unordered_map>
#include <vector>

static const char *_cudaGetErrorEnum(cufftResult error) {
  switch (error) {
  case CUFFT_SUCCESS:
    return "CUFFT_SUCCESS";

  case CUFFT_INVALID_PLAN:
    return "CUFFT_INVALID_PLAN";

  case CUFFT_ALLOC_FAILED:
    return "CUFFT_ALLOC_FAILED";

  case CUFFT_INVALID_TYPE:
    return "CUFFT_INVALID_TYPE";

  case CUFFT_INVALID_VALUE:
    return "CUFFT_INVALID_VALUE";

  case CUFFT_INTERNAL_ERROR:
    return "CUFFT_INTERNAL_ERROR";

  case CUFFT_EXEC_FAILED:
    return "CUFFT_EXEC_FAILED";

  case CUFFT_SETUP_FAILED:
    return "CUFFT_SETUP_FAILED";

  case CUFFT_INVALID_SIZE:
    return "CUFFT_INVALID_SIZE";

  case CUFFT_UNALIGNED_DATA:
    return "CUFFT_UNALIGNED_DATA";
  }

  return "<unknown>";
}
#define CUDA_CHECK(e)                                                          \
  {                                                                            \
    cudaError_t err = (e);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error: %s, line %d, %s: %s\n", __FILE__, __LINE__, \
              #e, cudaGetErrorString(err));                                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CUFFT_CHECK(e) { \
	cufftResult_t err = (e); \
	if (err != CUFFT_SUCCESS) \
	{ \
		fprintf(stderr, "CUFFT error: %s, line %d, %s: %s\n", \
			__FILE__, __LINE__, #e, _cudaGetErrorEnum(err)); \
		exit(EXIT_FAILURE); \
	} \
}

extern void *planWorkspace;

extern "C" void
#ifdef TRANS_SINGLE
execute_plan_fftc_(cufftHandle *PLANp, int *ISIGNp, cufftComplex *data_in,
                   cufftComplex *data_out)
#else
execute_plan_fftc_(cufftHandle *PLANp, int *ISIGNp, cufftDoubleComplex *data_in,
                   cufftDoubleComplex *data_out)
#endif
{
  cufftHandle plan = *PLANp;
  int ISIGN = *ISIGNp;

  CUFFT_CHECK(cufftSetWorkArea(plan, planWorkspace));

  if (ISIGN == -1) {
#ifdef TRANS_SINGLE
    CUFFT_CHECK(cufftExecR2C(plan, (cufftReal *)data_in, data_out));
#else
    CUFFT_CHECK(cufftExecD2Z(plan, (cufftDoubleReal *)data_in, data_out));
#endif
  } else if (ISIGN == 1) {
#ifdef TRANS_SINGLE
    CUFFT_CHECK(cufftExecC2R(plan, data_in, (cufftReal *)data_out));
#else
    CUFFT_CHECK(cufftExecZ2D(plan, data_in, (cufftDoubleReal *)data_out));
#endif
  } else {
    abort();
  }
}

namespace {
struct Double {
  using real = double;
  using cmplx = cufftDoubleComplex;
};
struct Float {
  using real = float;
  using cmplx = cufftComplex;
};
}
template <class Type, cufftType Direction>
void execute_fft(typename Type::real *data_real, typename Type::cmplx *data_complex,
        int kfield, int *loens, int *offsets, int nfft) {
  using real = typename Type::real;
  using cmplx = typename Type::cmplx;

  /* static std::unordered_map<int, void *> allocationCache; // nloens -> ptr */
  static std::unordered_map<int, std::vector<cufftHandle>> fftPlansCache; // kfield -> handles
  static std::unordered_map<int, cudaGraphExec_t> graphCache; // kfield -> graphs

  // if the pointers are changed, we need to update the graph
  static std::unordered_map<int, std::pair<real *, cmplx *>> ptrCache; // kfield -> ptrs

  auto ptrs = ptrCache.find(kfield);
  if (ptrs != ptrCache.end() && (
              ptrs->second.first != data_real || ptrs->second.second != data_complex)) {
      // the plan is cached, but the pointers are not correct. we remove and delete the graph,
      // but we keep the FFT plans, if this happens more often, we should cache this...
      std::cout << "WARNING: POINTER CHANGE --> THIS MIGHT BE SLOW" << std::endl;
      CUDA_CHECK(cudaGraphExecDestroy(graphCache[kfield]));
      graphCache.erase(kfield);
      ptrCache.erase(kfield);
  }

  auto graph = graphCache.find(kfield);
  if (graph == graphCache.end()) {
      // this graph does not exist yet

      auto fftPlans = fftPlansCache.find(kfield);
      if (fftPlans == fftPlansCache.end()) {
          // the fft plans do not exist yet
          std::vector<cufftHandle> newPlans;
          newPlans.resize(nfft);
          for (int i = 0; i < nfft; ++i) {
            int nloen = loens[i];

            cufftHandle plan;
            CUFFT_CHECK(cufftCreate(&plan));
            int dist = 1;
            int embed[] = {1};
            CUFFT_CHECK(cufftPlanMany(&plan, 1, &nloen, embed, kfield, dist, embed,
                                      kfield, dist, Direction, kfield));
            newPlans[i] = plan;
          }
          fftPlansCache.insert({kfield, newPlans});
      }
      fftPlans = fftPlansCache.find(kfield);

      // create a temporary stream
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreate(&stream));

      for (auto &plan : fftPlans->second) // set the streams
        CUFFT_CHECK(cufftSetStream(plan, stream));

      // now create the cuda graph
      cudaGraph_t new_graph;
      cudaGraphCreate(&new_graph, 0);
      for (int i = 0; i < nfft; ++i) {
        int offset = offsets[i];
        real *data_real_l = &data_real[kfield * offset];
        cmplx *data_complex_l = &data_complex[kfield * offset / 2];
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        if constexpr(Direction == CUFFT_R2C)
          CUFFT_CHECK(cufftExecR2C(fftPlans->second[i], data_real_l, data_complex_l))
        else if constexpr(Direction == CUFFT_C2R)
          CUFFT_CHECK(cufftExecC2R(fftPlans->second[i], data_complex_l, data_real_l))
        else if constexpr(Direction == CUFFT_D2Z)
          CUFFT_CHECK(cufftExecD2Z(fftPlans->second[i], data_real_l, data_complex_l))
        else if constexpr(Direction == CUFFT_Z2D)
          CUFFT_CHECK(cufftExecZ2D(fftPlans->second[i], data_complex_l, data_real_l));
        cudaGraph_t my_graph;
        CUDA_CHECK(cudaStreamEndCapture(stream, &my_graph));
        cudaGraphNode_t my_node;
        CUDA_CHECK(cudaGraphAddChildGraphNode(&my_node, new_graph, nullptr, 0, my_graph));
      }
      cudaGraphExec_t instance;
      CUDA_CHECK(cudaGraphInstantiate(&instance, new_graph, NULL, NULL, 0));
      CUDA_CHECK(cudaStreamDestroy(stream));
      CUDA_CHECK(cudaGraphDestroy(new_graph));

      graphCache.insert({kfield, instance});
      ptrCache.insert({kfield, std::make_pair(data_real, data_complex)});
  }

  CUDA_CHECK(cudaGraphLaunch(graphCache.at(kfield), 0));
      /* for (int i = 0; i < nfft; ++i) { */
        /* int nloen = loens[i]; */

        /* cufftHandle plan; */
        /* CUFFT_CHECK(cufftCreate(&plan)); */
        /* int dist = 1; */
        /* int embed[] = {1}; */
        /* CUFFT_CHECK(cufftPlanMany(&plan, 1, &nloen, embed, kfield, dist, embed, */
                                  /* kfield, dist, Direction, kfield)); */
        /* int offset = offsets[i]; */
        /* real *data_real_l = &data_real[kfield * offset]; */
        /* cmplx *data_complex_l = &data_complex[kfield * offset / 2]; */
        /* if (Direction == CUFFT_R2C) */
          /* CUFFT_CHECK(cufftExecR2C(plan, data_real_l, data_complex_l)) */
        /* else */
          /* CUFFT_CHECK(cufftExecC2R(plan, data_complex_l, data_real_l)); */
        /* CUFFT_CHECK(cufftDestroy(plan)); */
      /* } */
  CUDA_CHECK(cudaDeviceSynchronize());
}
template <class Type, cufftType Direction>
void execute_fft_new(typename Type::real *data_real, typename Type::cmplx *data_complex,
        int kfield, int *loens, int *offsets, int nfft) {
  using real = typename Type::real;
  using cmplx = typename Type::cmplx;

  /* static std::unordered_map<int, void *> allocationCache; // nloens -> ptr */
  static std::unordered_map<int, std::vector<cufftHandle>> fftPlansCache; // kfield -> handles
  static std::unordered_map<int, cudaGraphExec_t> graphCache; // kfield -> graphs

  // if the pointers are changed, we need to update the graph
  static std::unordered_map<int, std::pair<real *, cmplx *>> ptrCache; // kfield -> ptrs

  auto ptrs = ptrCache.find(kfield);
  if (ptrs != ptrCache.end() && (
              ptrs->second.first != data_real || ptrs->second.second != data_complex)) {
      // the plan is cached, but the pointers are not correct. we remove and delete the graph,
      // but we keep the FFT plans, if this happens more often, we should cache this...
      std::cout << "WARNING: POINTER CHANGE --> THIS MIGHT BE SLOW" << std::endl;
      CUDA_CHECK(cudaGraphExecDestroy(graphCache[kfield]));
      graphCache.erase(kfield);
      ptrCache.erase(kfield);
  }

  auto graph = graphCache.find(kfield);
  if (graph == graphCache.end()) {
      // this graph does not exist yet

      auto fftPlans = fftPlansCache.find(kfield);
      if (fftPlans == fftPlansCache.end()) {
          // the fft plans do not exist yet
          std::vector<cufftHandle> newPlans;
          newPlans.resize(nfft);
          for (int i = 0; i < nfft; ++i) {
            int nloen = loens[i];

            cufftHandle plan;
            CUFFT_CHECK(cufftCreate(&plan));
            int dist = offsets[i+1] - offsets[i];
            int embed[] = {1};
            CUFFT_CHECK(cufftPlanMany(&plan, 1, &nloen, embed, 1, dist, embed,
                                      1, dist / 2, Direction, kfield));
            newPlans[i] = plan;
          }
          fftPlansCache.insert({kfield, newPlans});
      }
      fftPlans = fftPlansCache.find(kfield);

      // create a temporary stream
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreate(&stream));

      for (auto &plan : fftPlans->second) // set the streams
        CUFFT_CHECK(cufftSetStream(plan, stream));

      // now create the cuda graph
      cudaGraph_t new_graph;
      cudaGraphCreate(&new_graph, 0);
      for (int i = 0; i < nfft; ++i) {
        int offset = offsets[i];
        real *data_real_l = &data_real[kfield * offset];
        cmplx *data_complex_l = &data_complex[kfield * offset / 2];
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        if constexpr(Direction == CUFFT_R2C)
          CUFFT_CHECK(cufftExecR2C(fftPlans->second[i], data_real_l, data_complex_l))
        else if constexpr(Direction == CUFFT_C2R)
          CUFFT_CHECK(cufftExecC2R(fftPlans->second[i], data_complex_l, data_real_l))
        else if constexpr(Direction == CUFFT_D2Z)
          CUFFT_CHECK(cufftExecD2Z(fftPlans->second[i], data_real_l, data_complex_l))
        else if constexpr(Direction == CUFFT_Z2D)
          CUFFT_CHECK(cufftExecZ2D(fftPlans->second[i], data_complex_l, data_real_l));
        cudaGraph_t my_graph;
        CUDA_CHECK(cudaStreamEndCapture(stream, &my_graph));
        cudaGraphNode_t my_node;
        CUDA_CHECK(cudaGraphAddChildGraphNode(&my_node, new_graph, nullptr, 0, my_graph));
      }
      cudaGraphExec_t instance;
      CUDA_CHECK(cudaGraphInstantiate(&instance, new_graph, NULL, NULL, 0));
      CUDA_CHECK(cudaStreamDestroy(stream));
      CUDA_CHECK(cudaGraphDestroy(new_graph));

      graphCache.insert({kfield, instance});
      ptrCache.insert({kfield, std::make_pair(data_real, data_complex)});
  }

  CUDA_CHECK(cudaGraphLaunch(graphCache.at(kfield), 0));
      /* for (int i = 0; i < nfft; ++i) { */
        /* int nloen = loens[i]; */

        /* cufftHandle plan; */
        /* CUFFT_CHECK(cufftCreate(&plan)); */
        /* int dist = 1; */
        /* int embed[] = {1}; */
        /* CUFFT_CHECK(cufftPlanMany(&plan, 1, &nloen, embed, kfield, dist, embed, */
                                  /* kfield, dist, Direction, kfield)); */
        /* int offset = offsets[i]; */
        /* real *data_real_l = &data_real[kfield * offset]; */
        /* cmplx *data_complex_l = &data_complex[kfield * offset / 2]; */
        /* if (Direction == CUFFT_R2C) */
          /* CUFFT_CHECK(cufftExecR2C(plan, data_real_l, data_complex_l)) */
        /* else */
          /* CUFFT_CHECK(cufftExecC2R(plan, data_complex_l, data_real_l)); */
        /* CUFFT_CHECK(cufftDestroy(plan)); */
      /* } */
  CUDA_CHECK(cudaDeviceSynchronize());
}
extern "C" {
void execute_dir_fft_float(float *data_real, cufftComplex *data_complex,
        int kfield, int *loens, int *offsets, int nfft) {
    execute_fft_new<Float, CUFFT_R2C>(data_real, data_complex, kfield, loens, offsets, nfft);
}
void execute_inv_fft_float(cufftComplex *data_complex, float *data_real,
        int kfield, int *loens, int *offsets, int nfft) {
    execute_fft<Float, CUFFT_C2R>(data_real, data_complex, kfield, loens, offsets, nfft);
}
void execute_dir_fft_double(double *data_real, cufftDoubleComplex *data_complex,
        int kfield, int *loens, int *offsets, int nfft) {
    execute_fft_new<Double, CUFFT_D2Z>(data_real, data_complex, kfield, loens, offsets, nfft);
}
void execute_inv_fft_double(cufftDoubleComplex *data_complex, double *data_real,
        int kfield, int *loens, int *offsets, int nfft) {
    execute_fft<Double, CUFFT_Z2D>(data_real, data_complex, kfield, loens, offsets, nfft);
}
}
