#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <numa.h>
#include <numaif.h>
#include <errno.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include <float.h>

#define NB_LOOPS 15

#define gettime(t) clock_gettime(CLOCK_MONOTONIC_RAW, t)
#define get_sub_seconde(t) (1e-9*(double)t.tv_nsec)

#ifdef __cplusplus
extern "C"
{
#endif
  /* return time in seconds */
  double get_elapsedtime(void);
  double get_elapsedtime_ns(void);
  double get_elapsedtime_us(void);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /* return time in second */
  double get_elapsedtime(void)
  {
    struct timespec st;
    int err = gettime(&st);
    if (err !=0) return 0;
    return (double)st.tv_sec + get_sub_seconde(st);
  }

  /* return time in nano seconds */
  double get_elapsedtime_ns(void)
  {
    struct timespec st;
    int err = gettime(&st);
    if (err !=0) return 0;
    return (double)st.tv_nsec;
  }

  /* return time in nano seconds */
  double get_elapsedtime_us(void)
  {
    struct timespec st;
    int err = gettime(&st);
    if (err !=0) return 0;
    return (double)st.tv_sec * 100000 + st.tv_nsec/10000;
  }
#ifdef __cplusplus
}
#endif


/*
 *  Allocate n bytes
 */
uint64_t* allocate(uint64_t size) {
  uint64_t* buf = (uint64_t*)mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, 0, 0);
  printf("allocate %lu byte\n", size);
  if(buf == MAP_FAILED)
  {
    perror("mmap");
    exit(1);
  }

  return buf;
}

/*
 *  Pine the current thread to the core core_id
 */
void pine_thread(int core_id) {
        cpu_set_t cpuset;

        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);

        if(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == -1) {
                perror("pthread_setaffinity_np");
                exit(1);
        }
}

/*
 *  Bind the nbc cache lines of the buffer the numa domain node_id
 */
void bind_memory(uint64_t* buf, uint64_t nbc, uint32_t node_id) {
  uint64_t nodemask = (1 << node_id);

  if(mbind(buf, nbc*sizeof(uint64_t),  MPOL_BIND, &nodemask, 64, MPOL_MF_MOVE) == -1) {
    perror("mbind");
    exit(1);
  }
}


/*
 *  Access all the nbc cache lines of the buffer buf
 */
void init_cpu(uint64_t* buf, uint64_t nbc, uint64_t val) {
        for(uint64_t i=0; i<nbc; i++)
	{
                buf[i] = val;
        }
}

__global__ void init_gpu(uint64_t* tab, uint64_t nbc, uint64_t val)
{
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if(tid < nbc)
    tab[tid] = val;
}

void addVec_cpu(uint64_t* c, uint64_t* a, uint64_t* b, uint64_t nbc)
{
  for(uint64_t i = 0; i < nbc; ++i)
  {
    c[i] = a[i] + b[i];
  }
}

__global__ void addVec_gpu(uint64_t* c, uint64_t* a, uint64_t* b, uint64_t nbc)
{
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if(tid < nbc)
    c[tid] = a[tid] + b[tid];
}

int main(int argc, char** argv)
{
  uint64_t size_b = 0;
  uint64_t size_mb = 0;
  int opt;
  int gpu_init = false;
  int bmem = -1;
  while ((opt = getopt(argc, argv, "hn:gb:")) != -1)
  {
    switch (opt)
    {
      case 'n':
        size_b = atoi(optarg) * 1024 * 1024;
        size_mb = atoi(optarg);
        break;
      case 'h':
        goto usage;
        break;
      case 'g':
        gpu_init = true; 
        break;
      case 'b':
	bmem = atoi(optarg);
	break;
      default:
        goto usage;
    }
  }

  if (optind != argc)
  {
usage:
    fprintf(stdout, "first-touch 1.0.0\n");
    fprintf(stdout, "usage: first-touch\n\t[-n size in MB]\n\t[-h print this help]\n");
    exit(EXIT_SUCCESS);
  }

  uint64_t nb_elem = size_b / sizeof(uint64_t);
  if(size_b%sizeof(uint64_t) !=0)
  {
    nb_elem++;
    size_b = nb_elem * sizeof(uint64_t);
  }

  //printf("Size: %" PRIu64 "B\n", size_b);
  printf("Size: %" PRIu64 "MB\n", size_mb);
  printf("Nb elems: %" PRIu64 "\n", nb_elem);

  int numa_node = -1;
  cudaSetDevice(0);
  dim3  dimBlock(64, 1, 1);
  dim3  dimGrid((nb_elem + dimBlock.x - 1)/dimBlock.x, 1, 1);

  pine_thread(0);

  uint64_t* a = allocate(size_b); 
  uint64_t* b = allocate(size_b); 
  uint64_t* c = allocate(size_b); 

  if(gpu_init)
  {
    init_gpu<<<dimGrid , dimBlock>>>(a, nb_elem, 0);
    init_gpu<<<dimGrid , dimBlock>>>(b, nb_elem, 1);
    init_gpu<<<dimGrid , dimBlock>>>(c, nb_elem, 2);
    for(uint64_t i=0; i<NB_LOOPS; i++) {     /* perform WARMUP NB_LOOPS */
      addVec_gpu<<<dimGrid, dimBlock>>>(c, a, b, nb_elem);
    }
  } else
  {
    init_cpu(a, nb_elem, 1);
    init_cpu(b, nb_elem, 2);
    init_cpu(c, nb_elem, 0);
    for(uint64_t i=0; i<NB_LOOPS; i++) {     /* perform WARMUP NB_LOOPS */
      addVec_cpu(c, a, b, nb_elem);
    }
  }

  numa_node = -1;
  get_mempolicy(&numa_node, NULL, 0, (void*)c, MPOL_F_NODE | MPOL_F_ADDR);
  printf("After Init: On node %d\n", numa_node);

  if(bmem != -1)
  {
    bind_memory(a, nb_elem, bmem);
    bind_memory(b, nb_elem, bmem);
    bind_memory(c, nb_elem, bmem);
    get_mempolicy(&numa_node, NULL, 0, (void*)c, MPOL_F_NODE | MPOL_F_ADDR);
    printf("After Binding: On node %d\n", numa_node);
  }

  printf("========\n\n");

  double *time_cpu = (double*) malloc(sizeof(double) * NB_LOOPS);
  double *time_gpu = (double*) malloc(sizeof(double) * NB_LOOPS);

  for(uint64_t i=0; i<NB_LOOPS; i++) {     /* perform NB_LOOPS */

    double start = get_elapsedtime_us();    /* start of experiment */
    addVec_gpu<<<dimGrid , dimBlock>>>(c, a, b, nb_elem);
    cudaDeviceSynchronize();
    double end = get_elapsedtime_us();      /* end of experiment */
    if(i != 0) printf("\n");
    printf("GPU> %.2lf ms to add an element (total: %.2lf ms)\n", (end - start)/(nb_elem), (end - start));
    time_gpu[i] = (end - start);

    start = get_elapsedtime_us();    /* start of experiment */
    addVec_cpu(c, a, b, nb_elem);                 /*   full access the buffer */
    end = get_elapsedtime_us();      /* end of experiment */
    printf("CPU> %.2lf ms to add an element (total: %.2lf ms)\n", (end - start)/(nb_elem), (end - start));
    time_cpu[i] = (end - start);
  }

  double *tmp = (double*) malloc(sizeof(double) * NB_LOOPS);
  double min = DBL_MAX;
  int idx = -1;

  for(int j = 0; j < NB_LOOPS; ++j)
  {
    for(int i = 0; i < NB_LOOPS; ++i)
    {
      if(time_gpu[i] < min)
      {
	min = time_gpu[i];
	idx = i;
      }
    }
    tmp[j] = min;
    time_gpu[idx] = DBL_MAX;

    min = DBL_MAX;
    idx = -1;
  }


  double *swap = tmp;
  tmp = time_gpu;
  time_gpu = swap;

  min = DBL_MAX;
  idx = -1;

  for(int j = 0; j < NB_LOOPS; ++j)
  {
    for(int i = 0; i < NB_LOOPS; ++i)
    {
      if(time_cpu[i] < min)
      {
	min = time_cpu[i];
	idx = i;
      }
    }
    tmp[j] = min;
    time_cpu[idx] = DBL_MAX;

    min = DBL_MAX;
    idx = -1;
  }

  swap = tmp;
  tmp = time_cpu;
  time_cpu = swap;

  int med = (NB_LOOPS / 2) + 1;
  printf("Median (GPU): %.2lf\nMedian (CPU): %.2lf\n", time_gpu[med], time_cpu[med]);

  munmap(a, size_b);
  munmap(b, size_b);
  munmap(c, size_b);

  free(time_cpu);
  free(time_gpu);

  return 0;
}
