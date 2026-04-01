/* findall_v3.c
 *
 * Trova gli indici di tutti gli elementi positivi in un array di char
 * usando una pipeline OpenCL con prefix-sum globale Kogge-Stone.
 *
 * Pipeline:
 *   1. mark_positive_kernel : scrive 0/1 in d_scan[ping]
 *   2. scan_stride_kernel   : lanciato ceil(log2(nels)) volte con ping-pong
 *                             tra d_scan[0] e d_scan[1]
 *   3. Leggi count = d_scan[risultato][nels-1]
 *   4. findall_gpu          : scatter degli indici
 *
 * La scan e' GLOBALE: non ci sono block sums, scan_sums o apply_offsets.
 * Ogni lancio di kernel e' la barriera globale tra i passi della scan.
 * Funziona correttamente su GPU reale e su PoCL/CPU.
 */

#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

void error(const char *err) {
  fprintf(stderr, "%s\n", err);
  exit(1);
}

/* Fase 1: marca gli elementi positivi scrivendo 0/1 in d_scan_ping */
cl_event mark_positive(cl_command_queue que, cl_kernel mark_kernel,
                       cl_mem d_scan_ping, cl_mem d_input, int nels,
                       size_t lws) {
  size_t gws = round_mul_up(nels, lws);
  printf("mark_positive: gws=%zu lws=%zu\n", gws, lws);

  cl_int err;
  cl_event ret;
  int arg = 0;
  err = clSetKernelArg(mark_kernel, arg++, sizeof(d_scan_ping), &d_scan_ping);
  ocl_check(err, "mark arg scan_ping");
  err = clSetKernelArg(mark_kernel, arg++, sizeof(d_input), &d_input);
  ocl_check(err, "mark arg input");
  err = clSetKernelArg(mark_kernel, arg++, sizeof(nels), &nels);
  ocl_check(err, "mark arg nels");

  err = clEnqueueNDRangeKernel(que, mark_kernel, 1, NULL, &gws, &lws, 0, NULL,
                               &ret);
  ocl_check(err, "enqueue mark_positive");
  return ret;
}

/* Un passo della scan: legge da src, scrive su dst */
cl_event scan_stride(cl_command_queue que, cl_kernel scan_kernel,
                     cl_event wait_for, cl_mem dst, cl_mem src, int stride,
                     int nels, size_t lws) {
  size_t gws = round_mul_up(nels, lws);

  cl_int err;
  cl_event ret;
  int arg = 0;
  err = clSetKernelArg(scan_kernel, arg++, sizeof(dst), &dst);
  ocl_check(err, "scan_stride arg dst");
  err = clSetKernelArg(scan_kernel, arg++, sizeof(src), &src);
  ocl_check(err, "scan_stride arg src");
  err = clSetKernelArg(scan_kernel, arg++, sizeof(stride), &stride);
  ocl_check(err, "scan_stride arg stride");
  err = clSetKernelArg(scan_kernel, arg++, sizeof(nels), &nels);
  ocl_check(err, "scan_stride arg nels");

  err = clEnqueueNDRangeKernel(que, scan_kernel, 1, NULL, &gws, &lws, 1,
                               &wait_for, &ret);
  ocl_check(err, "enqueue scan_stride");
  return ret;
}

/* Fase 3: scatter degli indici positivi nell'output */
cl_event findall_final(cl_command_queue que, cl_kernel findall_kernel,
                       cl_event wait_for, cl_mem d_output, cl_mem d_flags,
                       cl_mem d_scan, int nels, size_t lws) {
  size_t gws = round_mul_up(nels, lws);
  printf("findall_final: gws=%zu lws=%zu\n", gws, lws);

  cl_int err;
  cl_event ret;
  int arg = 0;
  err = clSetKernelArg(findall_kernel, arg++, sizeof(d_output), &d_output);
  ocl_check(err, "findall arg output");
  err = clSetKernelArg(findall_kernel, arg++, sizeof(d_flags), &d_flags);
  ocl_check(err, "findall arg flags");
  err = clSetKernelArg(findall_kernel, arg++, sizeof(d_scan), &d_scan);
  ocl_check(err, "findall arg scan");
  err = clSetKernelArg(findall_kernel, arg++, sizeof(nels), &nels);
  ocl_check(err, "findall arg nels");

  err = clEnqueueNDRangeKernel(que, findall_kernel, 1, NULL, &gws, &lws, 1,
                               &wait_for, &ret);
  ocl_check(err, "enqueue findall_final");
  return ret;
}

void verify_result(const cl_int *output, int count, const char *input,
                   int nels) {
  printf("Found %d positive elements", count);

  if (count <= 20 && count > 0) {
    printf(" at indices: ");
    for (int i = 0; i < count; i++)
      printf("%d ", output[i]);
  } else if (count > 0) {
    printf(" at indices (first 10): ");
    for (int i = 0; i < 10; i++)
      printf("%d ", output[i]);
    printf("... (%d more)", count - 10);
  }
  printf("\n");

  int expected = 0;
  for (int i = 0; i < nels; i++)
    if (input[i] > 0)
      expected++;

  if (expected != count)
    fprintf(stderr, "FAILED: expected %d, got %d\n", expected, count);
  else
    printf("Verification: PASSED\n");
}

int main(int argc, char *argv[]) {
  if (argc < 3)
    error("usage: findall <lws> <nels>");

  int lws_i = atoi(argv[1]);
  if (lws_i <= 0)
    error("lws must be positive");
  if (lws_i > 1024) {
    lws_i = 1024;
    fprintf(stderr, "lws clamped to 1024\n");
  }
  size_t lws = (size_t)lws_i;

  long long nels_ll = atoll(argv[2]);
  if (nels_ll <= 0)
    error("nels must be positive");
  if (nels_ll > INT_MAX)
    error("nels too large");
  int nels = (int)nels_ll;

  /* Genera dati di test */
  srand(42);
  char *vals = malloc(nels * sizeof(char));
  if (!vals)
    error("malloc failed");
  for (int i = 0; i < nels; i++)
    vals[i] = (char)(rand() % 5 - 2);

  printf("Array size: %d\n", nels);
  if (nels <= 20) {
    printf("Input: ");
    for (int i = 0; i < nels; i++)
      printf("%d ", vals[i]);
    printf("\n");
  }

  /* Numero di passi della scan: ceil(log2(nels)) */
  int num_strides = 0;
  {
    int tmp = nels - 1;
    while (tmp > 0) {
      num_strides++;
      tmp >>= 1;
    }
  }
  if (num_strides == 0)
    num_strides = 1;
  printf("Scan strides: %d\n", num_strides);

  /* Setup OpenCL */
  cl_platform_id p = select_platform();
  cl_device_id d = select_device(p);
  cl_context ctx = create_context(p, d);
  cl_command_queue que = create_queue(ctx, d);
  cl_program prog = create_program("findall_final.ocl", ctx, d);

  cl_int err;

  cl_kernel mark_kernel = clCreateKernel(prog, "mark_positive_kernel", &err);
  ocl_check(err, "create mark_positive_kernel");
  cl_kernel scan_kernel = clCreateKernel(prog, "scan_stride_kernel", &err);
  ocl_check(err, "create scan_stride_kernel");
  cl_kernel findall_kernel = clCreateKernel(prog, "findall_gpu", &err);
  ocl_check(err, "create findall_gpu");

  size_t scan_memsize = nels * sizeof(cl_int);

  /* d_input: dati originali (char) */
  cl_mem d_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  nels * sizeof(char), vals, &err);
  ocl_check(err, "create d_input");

  /* d_flags: copia permanente dei flag 0/1 per la fase findall */
  cl_mem d_flags =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, scan_memsize, NULL, &err);
  ocl_check(err, "create d_flags");

  /* Due buffer per il ping-pong della scan globale */
  cl_mem d_scan[2];
  d_scan[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, scan_memsize, NULL, &err);
  ocl_check(err, "create d_scan[0]");
  d_scan[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, scan_memsize, NULL, &err);
  ocl_check(err, "create d_scan[1]");

  /* ------------------------------------------------------------------ */
  /* Fase 1: mark -> scrive 0/1 in d_scan[0] e in d_flags              */
  /* ------------------------------------------------------------------ */
  printf("\nPhase 1: mark_positive\n");
  cl_event mark_evt =
      mark_positive(que, mark_kernel, d_scan[0], d_input, nels, lws);

  /* Copia d_scan[0] in d_flags (servirà per findall_gpu) */
  cl_event copy_evt;
  err = clEnqueueCopyBuffer(que, d_scan[0], d_flags, 0, 0, scan_memsize, 1,
                            &mark_evt, &copy_evt);
  ocl_check(err, "copy d_scan[0] -> d_flags");

  /* ------------------------------------------------------------------ */
  /* Fase 2: scan globale con ping-pong                                  */
  /* ------------------------------------------------------------------ */
  printf("Phase 2: global prefix-sum (%d strides)\n", num_strides);

  int ping = 0;
  cl_event last_stride_evt = NULL;
  cl_event prev_evt = copy_evt;

  for (int s = 0; s < num_strides; s++) {
    int stride = 1 << s;
    int src_buf = ping;
    int dst_buf = 1 - ping;

    printf("  stride=%d: d_scan[%d] -> d_scan[%d]\n", stride, src_buf, dst_buf);

    cl_event stride_evt =
        scan_stride(que, scan_kernel, prev_evt, d_scan[dst_buf],
                    d_scan[src_buf], stride, nels, lws);

    /* Rilascia il precedente stride (non copy_evt che e' rilasciato nel
     * cleanup) */
    if (last_stride_evt != NULL)
      clReleaseEvent(last_stride_evt);

    last_stride_evt = stride_evt;
    prev_evt = stride_evt;
    ping = dst_buf;
  }

  cl_mem d_scan_result = d_scan[ping];
  cl_event scan_done_evt =
      (last_stride_evt != NULL) ? last_stride_evt : copy_evt;

  /* ------------------------------------------------------------------ */
  /* Leggi il count                                                       */
  /* ------------------------------------------------------------------ */
  cl_event map_evt;
  cl_int *h_scan =
      clEnqueueMapBuffer(que, d_scan_result, CL_TRUE, CL_MAP_READ, 0,
                         scan_memsize, 1, &scan_done_evt, &map_evt, &err);
  ocl_check(err, "map scan_result");

  int count = h_scan[nels - 1];
  printf("Total positive elements: %d\n", count);

  clEnqueueUnmapMemObject(que, d_scan_result, h_scan, 0, NULL, NULL);
  clReleaseEvent(map_evt);

  if (count < 0 || count > nels) {
    fprintf(stderr, "ERROR: count %d out of range [0,%d]\n", count, nels);
    goto cleanup;
  }

  /* ------------------------------------------------------------------ */
  /* Fase 3: findall scatter                                              */
  /* ------------------------------------------------------------------ */
  if (count > 0) {
    size_t out_size = count * sizeof(cl_int);
    cl_mem d_output = clCreateBuffer(
        ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, out_size, NULL, &err);
    ocl_check(err, "create d_output");

    printf("\nPhase 3: findall scatter\n");
    cl_event findall_evt =
        findall_final(que, findall_kernel, scan_done_evt, d_output, d_flags,
                      d_scan_result, nels, lws);

    err = clWaitForEvents(1, &findall_evt);
    ocl_check(err, "wait findall");

    cl_event map_out_evt;
    cl_int *h_output =
        clEnqueueMapBuffer(que, d_output, CL_TRUE, CL_MAP_READ, 0, out_size, 1,
                           &findall_evt, &map_out_evt, &err);
    ocl_check(err, "map output");

    verify_result(h_output, count, vals, nels);

    double mark_ms = runtime_ms(mark_evt);
    double findall_ms = runtime_ms(findall_evt);
    double total_ms = total_runtime_ms(mark_evt, findall_evt);

    printf("\nPerformance:\n");
    printf("  mark:    %.3f ms\n", mark_ms);
    printf("  findall: %.3f ms\n", findall_ms);
    printf("  total:   %.3f ms  (%.2f GE/s)\n", total_ms,
           nels / total_ms / 1.0e6);

    clEnqueueUnmapMemObject(que, d_output, h_output, 0, NULL, NULL);
    clReleaseEvent(map_out_evt);
    clReleaseEvent(findall_evt);
    clReleaseMemObject(d_output);
  }

cleanup:
  err = clFinish(que);
  ocl_check(err, "finish");

  /* Ogni evento viene rilasciato esattamente una volta */
  clReleaseEvent(mark_evt);
  clReleaseEvent(copy_evt);
  if (last_stride_evt != NULL)
    clReleaseEvent(last_stride_evt);

  clReleaseKernel(mark_kernel);
  clReleaseKernel(scan_kernel);
  clReleaseKernel(findall_kernel);
  clReleaseMemObject(d_input);
  clReleaseMemObject(d_flags);
  clReleaseMemObject(d_scan[0]);
  clReleaseMemObject(d_scan[1]);
  clReleaseProgram(prog);
  clReleaseCommandQueue(que);
  clReleaseContext(ctx);
  free(vals);
  return 0;
}