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

#define NB_LOOPS 51

#define gettime(t) clock_gettime(CLOCK_MONOTONIC_RAW, t)
#define get_sub_seconde(t) (1e-9*(double)t.tv_nsec)

#ifdef __cplusplus
extern "C"
{
#endif
  /* return time in seconds */
  double get_elapsedtime(void);
  double get_elapsedtime_ns(void);
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
 *  Prepare the buffer to perform a randomized access
 */
void randomize(uint64_t* buf, uint64_t n) {
        uint64_t c = 0, r;

        srand(time(NULL));

        for(uint64_t i=0; i<(n-1); i++) {
                buf[c] = 1;

                do {
                        r = rand() % n;
                } while(buf[r]);

                buf[c] = r;

                c = r;
        }

        buf[c] = 0;
}

/*
 *  Access all the nbc cache lines of the buffer buf
 */
uint64_t rnd_access(uint64_t* buf, uint64_t nbc) {
        uint64_t r = 0;
        for(uint64_t i=0; i<nbc; i++)
	{
                r = buf[r];
        }
        return r;
}

uint64_t rnd_access_first(uint64_t* buf, uint64_t* first, uint64_t nbc) {
        uint64_t r = 0;
        for(uint64_t i=0; i<nbc; i++)
	{
                r = buf[r];
		first[i] = r;
        }
        return r;
}

__global__ void rnd_access_gpu(uint64_t* buf, uint64_t* tab, uint64_t nbc)
{
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  uint64_t r = 0;
  if(tid < nbc)
    r = buf[ tab[tid] ];
  r++;
}

__global__ void init(uint64_t* tab, uint64_t nbc)
{
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if(tid < nbc)
    tab[tid] = 0;
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
        size_b = atoi(optarg) *1024*1024;
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

  pine_thread(0);
  cudaSetDevice(0);

  dim3  dimBlock(64, 1, 1);
  dim3  dimGrid((nb_elem + dimBlock.x - 1)/dimBlock.x, 1, 1);

  uint64_t* tab = allocate(size_b); 

  if(gpu_init)
  {
    init<<<dimGrid , dimBlock>>>(tab, nb_elem);
  } else
  {
    memset(tab, 0, size_b);
  }

  if(bmem != -1)
  {
    printf("Binding on node... %d\n", bmem);
    bind_memory(tab, nb_elem, bmem);
    get_mempolicy(&numa_node, NULL, 0, (void*)tab, MPOL_F_NODE | MPOL_F_ADDR);
    printf("After Binding: On node %d\n", numa_node);
  }

  randomize(tab, nb_elem);

  uint64_t* tab2 = allocate(size_b); 
  rnd_access_first(tab, tab2, nb_elem);

  uint64_t* tab_gpu;
  cudaMalloc(&tab_gpu, size_b);
  cudaMemcpy(tab_gpu, tab2, size_b, cudaMemcpyHostToDevice);

  if(gpu_init)
  {
    for(uint64_t i=0; i<NB_LOOPS; i++) {     /* perform NB_LOOPS */
      rnd_access_gpu<<<dimGrid , dimBlock>>>(tab, tab_gpu, nb_elem);
    }
  } else
  {
    for(uint64_t i=0; i<NB_LOOPS; i++) {     /* perform NB_LOOPS */
      rnd_access(tab, nb_elem);
    }
  }

  numa_node = -1;
  get_mempolicy(&numa_node, NULL, 0, (void*)tab, MPOL_F_NODE | MPOL_F_ADDR);
  printf("After Init: On node %d\n========\n\n", numa_node);
  uint64_t r = 0;

  for(uint64_t i=0; i<NB_LOOPS; i++) {     /* perform NB_LOOPS */
    if(i != 0) printf("\n");
    printf("=== %d ===\n", (int)i);
    double start = get_elapsedtime_ns();    /* start of experiment */
    rnd_access_gpu<<<dimGrid , dimBlock>>>(tab, tab_gpu, nb_elem);
    double end = get_elapsedtime_ns();      /* end of experiment */
    printf("GPU> %.2lf ns to access an element (total: %.2lf ns)\n", (end - start)/(nb_elem), (end - start));

    start = get_elapsedtime_ns();    /* start of experiment */
    r += rnd_access(tab, nb_elem);                 /*   full access the buffer */
    end = get_elapsedtime_ns();      /* end of experiment */
    printf("CPU> %.2lf ns to access an element (total: %.2lf ns)\n", (end - start)/(nb_elem), (end - start));

  }


  munmap(tab , nb_elem);
  munmap(tab2, nb_elem);
  cudaFree(tab_gpu);

  return 0;
}
