/* findall_lmem.c
 *
 * Scan ricorsiva a 3 livelli — completamente parallela, nessun limite pratico.
 *
 * Con lws=256:  max nels = 256^3 = 16M
 * Con lws=1024: max nels = 1024^3 = 1G
 *
 * Pipeline (6 kernel, O(N) traffico memoria):
 *
 *   nels elementi
 *   └─ scan_local_k0 → scan0[nels], sums0[ng0]   ng0 = ceil(nels/lws)
 *      └─ scan_local_k1 → scan1[ng0], sums1[ng1]  ng1 = ceil(ng0/lws)
 *         └─ scan_local_k2 → offsets1[ng1]         (singolo work-group)
 *            └─ apply_k1 → scan1_global[ng0]
 *               └─ apply_k0 → scan_global[nels]
 *                  └─ findall_gpu → output[count]
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

/* --- helper: setta args e lancia un kernel 1D ----------------------------- */
static cl_event launch1d(cl_command_queue que, cl_kernel k, size_t gws,
                         size_t lws, cl_uint nwait, cl_event *wait_list) {
  cl_event ret;
  cl_int err = clEnqueueNDRangeKernel(que, k, 1, NULL, &gws, &lws, nwait,
                                      nwait ? wait_list : NULL, &ret);
  ocl_check(err, "launch1d");
  return ret;
}

/* --- wrappers ------------------------------------------------------------- */

cl_event run_scan_k0(cl_command_queue que, cl_kernel k, cl_mem d_flags,
                     cl_mem d_scan0, cl_mem d_sums0, cl_mem d_input, int nels,
                     size_t lws) {
  int arg = 0;
  cl_int err;
  err = clSetKernelArg(k, arg++, sizeof(d_flags), &d_flags);
  ocl_check(err, "k0 flags");
  err = clSetKernelArg(k, arg++, sizeof(d_scan0), &d_scan0);
  ocl_check(err, "k0 scan0");
  err = clSetKernelArg(k, arg++, sizeof(d_sums0), &d_sums0);
  ocl_check(err, "k0 sums0");
  err = clSetKernelArg(k, arg++, sizeof(d_input), &d_input);
  ocl_check(err, "k0 input");
  err = clSetKernelArg(k, arg++, sizeof(nels), &nels);
  ocl_check(err, "k0 nels");
  err = clSetKernelArg(k, arg++, lws * sizeof(cl_int), NULL);
  ocl_check(err, "k0 lmem");
  size_t gws = round_mul_up(nels, lws);
  return launch1d(que, k, gws, lws, 0, NULL);
}

cl_event run_scan_k1(cl_command_queue que, cl_kernel k, cl_event wait,
                     cl_mem d_scan1, cl_mem d_sums1, cl_mem d_sums0, int n1,
                     size_t lws) {
  int arg = 0;
  cl_int err;
  err = clSetKernelArg(k, arg++, sizeof(d_scan1), &d_scan1);
  ocl_check(err, "k1 scan1");
  err = clSetKernelArg(k, arg++, sizeof(d_sums1), &d_sums1);
  ocl_check(err, "k1 sums1");
  err = clSetKernelArg(k, arg++, sizeof(d_sums0), &d_sums0);
  ocl_check(err, "k1 sums0");
  err = clSetKernelArg(k, arg++, sizeof(n1), &n1);
  ocl_check(err, "k1 n1");
  err = clSetKernelArg(k, arg++, lws * sizeof(cl_int), NULL);
  ocl_check(err, "k1 lmem");
  size_t gws = round_mul_up(n1, lws);
  return launch1d(que, k, gws, lws, 1, &wait);
}

cl_event run_scan_k2(cl_command_queue que, cl_kernel k, cl_event wait,
                     cl_mem d_offsets1, cl_mem d_sums1, int n2, size_t lws) {
  int arg = 0;
  cl_int err;
  err = clSetKernelArg(k, arg++, sizeof(d_offsets1), &d_offsets1);
  ocl_check(err, "k2 off");
  err = clSetKernelArg(k, arg++, sizeof(d_sums1), &d_sums1);
  ocl_check(err, "k2 sums1");
  err = clSetKernelArg(k, arg++, sizeof(n2), &n2);
  ocl_check(err, "k2 n2");
  err = clSetKernelArg(k, arg++, lws * sizeof(cl_int), NULL);
  ocl_check(err, "k2 lmem");
  size_t gws = round_mul_up(n2, lws);
  return launch1d(que, k, gws, lws, 1, &wait);
}

cl_event run_apply_k1(cl_command_queue que, cl_kernel k, cl_event wait,
                      cl_mem d_scan1g, cl_mem d_scan1, cl_mem d_offsets1,
                      int n1, size_t lws) {
  int arg = 0;
  cl_int err;
  err = clSetKernelArg(k, arg++, sizeof(d_scan1g), &d_scan1g);
  ocl_check(err, "a1 sg");
  err = clSetKernelArg(k, arg++, sizeof(d_scan1), &d_scan1);
  ocl_check(err, "a1 s1");
  err = clSetKernelArg(k, arg++, sizeof(d_offsets1), &d_offsets1);
  ocl_check(err, "a1 off");
  err = clSetKernelArg(k, arg++, sizeof(n1), &n1);
  ocl_check(err, "a1 n1");
  size_t gws = round_mul_up(n1, lws);
  return launch1d(que, k, gws, lws, 1, &wait);
}

cl_event run_apply_k0(cl_command_queue que, cl_kernel k, cl_event wait,
                      cl_mem d_scan_global, cl_mem d_scan0, cl_mem d_scan1g,
                      int nels, size_t lws) {
  int arg = 0;
  cl_int err;
  err = clSetKernelArg(k, arg++, sizeof(d_scan_global), &d_scan_global);
  ocl_check(err, "a0 sg");
  err = clSetKernelArg(k, arg++, sizeof(d_scan0), &d_scan0);
  ocl_check(err, "a0 s0");
  err = clSetKernelArg(k, arg++, sizeof(d_scan1g), &d_scan1g);
  ocl_check(err, "a0 s1g");
  err = clSetKernelArg(k, arg++, sizeof(nels), &nels);
  ocl_check(err, "a0 nels");
  size_t gws = round_mul_up(nels, lws);
  return launch1d(que, k, gws, lws, 1, &wait);
}

cl_event run_findall(cl_command_queue que, cl_kernel k, cl_event wait,
                     cl_mem d_output, cl_mem d_flags, cl_mem d_scan_global,
                     int nels, size_t lws) {
  int arg = 0;
  cl_int err;
  err = clSetKernelArg(k, arg++, sizeof(d_output), &d_output);
  ocl_check(err, "f out");
  err = clSetKernelArg(k, arg++, sizeof(d_flags), &d_flags);
  ocl_check(err, "f flags");
  err = clSetKernelArg(k, arg++, sizeof(d_scan_global), &d_scan_global);
  ocl_check(err, "f scan");
  err = clSetKernelArg(k, arg++, sizeof(nels), &nels);
  ocl_check(err, "f nels");
  size_t gws = round_mul_up(nels, lws);
  return launch1d(que, k, gws, lws, 1, &wait);
}

/* --- verifica ------------------------------------------------------------- */

void verify_result(const cl_int *out, int count, const char *input, int nels) {
  printf("Found %d positive elements", count);
  if (count > 0 && count <= 20) {
    printf(" at indices:");
    for (int i = 0; i < count; i++)
      printf(" %d", out[i]);
  } else if (count > 0) {
    printf(" at indices (first 10):");
    for (int i = 0; i < 10; i++)
      printf(" %d", out[i]);
    printf(" ... (%d more)", count - 10);
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

/* --- main ----------------------------------------------------------------- */

int main(int argc, char *argv[]) {
  if (argc < 3)
    error("usage: findall_lmem <lws> <nels>");

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

  /* Calcola dimensioni dei livelli */
  int ng0 = round_div_up(nels, lws_i); /* num blocchi livello 0 */
  int ng1 = round_div_up(ng0, lws_i);  /* num blocchi livello 1 */
  int ng2 = round_div_up(ng1, lws_i);  /* deve essere 1         */

  if (ng2 > lws_i) {
    fprintf(stderr,
            "ERROR: array troppo grande per 3 livelli con lws=%d.\n"
            "       Max nels = %lld. Usa lws piu' grande.\n",
            lws_i, (long long)lws_i * lws_i * lws_i);
    exit(1);
  }

  printf("Array size: %d  lws: %zu\n", nels, lws);
  printf("Livelli: ng0=%d  ng1=%d  ng2=%d\n", ng0, ng1, ng2);

  srand(42);
  char *vals = malloc(nels * sizeof(char));
  if (!vals)
    error("malloc failed");
  for (int i = 0; i < nels; i++)
    vals[i] = (char)(rand() % 5 - 2);

  if (nels <= 20) {
    printf("Input:");
    for (int i = 0; i < nels; i++)
      printf(" %d", vals[i]);
    printf("\n");
  }

  /* OpenCL setup */
  cl_platform_id p = select_platform();
  cl_device_id d = select_device(p);

  /* Clamp lws to the device's actual maximum work-group size */
  size_t max_lws = 0;
  clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_lws), &max_lws,
                  NULL);
  if (lws > max_lws) {
    fprintf(stderr,
            "lws clamped from %zu to %zu (CL_DEVICE_MAX_WORK_GROUP_SIZE)\n",
            lws, max_lws);
    lws = max_lws;
    lws_i = (int)max_lws;
    /* Recompute level sizes with the new lws */
    ng0 = round_div_up(nels, lws_i);
    ng1 = round_div_up(ng0, lws_i);
    ng2 = round_div_up(ng1, lws_i);
    printf("Livelli (after clamp): ng0=%d  ng1=%d  ng2=%d\n", ng0, ng1, ng2);
  }

  cl_context ctx = create_context(p, d);
  cl_command_queue que = create_queue(ctx, d);
  cl_program prog = create_program("findall_lmem_v2.ocl", ctx, d);
  cl_int err;

  cl_kernel k0 = clCreateKernel(prog, "scan_local_k0", &err);
  ocl_check(err, "k0");
  cl_kernel k1 = clCreateKernel(prog, "scan_local_k1", &err);
  ocl_check(err, "k1");
  cl_kernel k2 = clCreateKernel(prog, "scan_local_k2", &err);
  ocl_check(err, "k2");
  cl_kernel ka1 = clCreateKernel(prog, "apply_k1", &err);
  ocl_check(err, "ka1");
  cl_kernel ka0 = clCreateKernel(prog, "apply_k0", &err);
  ocl_check(err, "ka0");
  cl_kernel kfind = clCreateKernel(prog, "findall_gpu", &err);
  ocl_check(err, "kfind");

  /* Alloca buffer */
  cl_mem d_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  nels * sizeof(char), vals, &err);
  ocl_check(err, "input");
  cl_mem d_flags =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, nels * sizeof(cl_int), NULL, &err);
  ocl_check(err, "flags");
  cl_mem d_scan0 =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, nels * sizeof(cl_int), NULL, &err);
  ocl_check(err, "scan0");
  cl_mem d_sums0 =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, ng0 * sizeof(cl_int), NULL, &err);
  ocl_check(err, "sums0");
  cl_mem d_scan1 =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, ng0 * sizeof(cl_int), NULL, &err);
  ocl_check(err, "scan1");
  cl_mem d_sums1 =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, ng1 * sizeof(cl_int), NULL, &err);
  ocl_check(err, "sums1");
  cl_mem d_offsets1 =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, ng1 * sizeof(cl_int), NULL, &err);
  ocl_check(err, "offsets1");
  cl_mem d_scan1g =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, ng0 * sizeof(cl_int), NULL, &err);
  ocl_check(err, "scan1g");
  cl_mem d_scan_global =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, nels * sizeof(cl_int), NULL, &err);
  ocl_check(err, "scan_global");

  /* ------------------------------------------------------------------ */
  /* Pipeline                                                             */
  /* ------------------------------------------------------------------ */
  cl_event e0 =
      run_scan_k0(que, k0, d_flags, d_scan0, d_sums0, d_input, nels, lws);

  cl_event e1 = run_scan_k1(que, k1, e0, d_scan1, d_sums1, d_sums0, ng0, lws);

  cl_event e2 = run_scan_k2(que, k2, e1, d_offsets1, d_sums1, ng1, lws);

  cl_event ea1 =
      run_apply_k1(que, ka1, e2, d_scan1g, d_scan1, d_offsets1, ng0, lws);

  cl_event ea0 =
      run_apply_k0(que, ka0, ea1, d_scan_global, d_scan0, d_scan1g, nels, lws);

  /* Leggi count */
  cl_event e_map;
  cl_int *h_scan =
      clEnqueueMapBuffer(que, d_scan_global, CL_TRUE, CL_MAP_READ, 0,
                         nels * sizeof(cl_int), 1, &ea0, &e_map, &err);
  ocl_check(err, "map scan_global");
  int count = h_scan[nels - 1];
  printf("Total positive elements: %d\n", count);
  clEnqueueUnmapMemObject(que, d_scan_global, h_scan, 0, NULL, NULL);
  clReleaseEvent(e_map);

  if (count < 0 || count > nels) {
    fprintf(stderr, "ERROR: count %d out of range\n", count);
    goto cleanup;
  }

  if (count > 0) {
    size_t out_sz = count * sizeof(cl_int);
    cl_mem d_output = clCreateBuffer(
        ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, out_sz, NULL, &err);
    ocl_check(err, "d_output");

    cl_event e_find = run_findall(que, kfind, ea0, d_output, d_flags,
                                  d_scan_global, nels, lws);
    err = clWaitForEvents(1, &e_find);
    ocl_check(err, "wait e_find");

    cl_event e_mout;
    cl_int *h_out = clEnqueueMapBuffer(que, d_output, CL_TRUE, CL_MAP_READ, 0,
                                       out_sz, 1, &e_find, &e_mout, &err);
    ocl_check(err, "map output");

    verify_result(h_out, count, vals, nels);

    /* Performance */
    double ms0 = runtime_ms(e0);
    double ms1 = runtime_ms(e1);
    double ms2 = runtime_ms(e2);
    double msa1 = runtime_ms(ea1);
    double msa0 = runtime_ms(ea0);
    double msfind = runtime_ms(e_find);
    double mstot = total_runtime_ms(e0, e_find);

    size_t sc = nels * sizeof(cl_int);
    size_t s0 = ng0 * sizeof(cl_int);
    size_t s1 = ng1 * sizeof(cl_int);
    size_t ob = count * sizeof(cl_int);

    printf("\nPerformance:\n");
    printf("  scan_k0:   %7.3f ms  bw: %.2f GB/s\n", ms0,
           (nels * sizeof(char) + 2 * sc + s0) / ms0 / 1e6);
    printf("  scan_k1:   %7.3f ms  bw: %.2f GB/s\n", ms1,
           (s0 + 2 * s0 + s1) / ms1 / 1e6);
    printf("  scan_k2:   %7.3f ms  bw: %.2f GB/s\n", ms2,
           (2.0 * s1) / ms2 / 1e6);
    printf("  apply_k1:  %7.3f ms  bw: %.2f GB/s\n", msa1,
           (2.0 * s0 + s1) / msa1 / 1e6);
    printf("  apply_k0:  %7.3f ms  bw: %.2f GB/s\n", msa0,
           (2.0 * sc + s0) / msa0 / 1e6);
    printf("  findall:   %7.3f ms  bw: %.2f GB/s\n", msfind,
           (sc + sc + ob) / msfind / 1e6);
    printf("  TOTAL:     %7.3f ms  throughput: %.2f GE/s\n", mstot,
           nels / mstot / 1e6);

    clEnqueueUnmapMemObject(que, d_output, h_out, 0, NULL, NULL);
    clReleaseEvent(e_mout);
    clReleaseEvent(e_find);
    clReleaseMemObject(d_output);
  }

cleanup:
  clFinish(que);
  clReleaseEvent(e0);
  clReleaseEvent(e1);
  clReleaseEvent(e2);
  clReleaseEvent(ea1);
  clReleaseEvent(ea0);
  clReleaseKernel(k0);
  clReleaseKernel(k1);
  clReleaseKernel(k2);
  clReleaseKernel(ka1);
  clReleaseKernel(ka0);
  clReleaseKernel(kfind);
  clReleaseMemObject(d_input);
  clReleaseMemObject(d_flags);
  clReleaseMemObject(d_scan0);
  clReleaseMemObject(d_sums0);
  clReleaseMemObject(d_scan1);
  clReleaseMemObject(d_sums1);
  clReleaseMemObject(d_offsets1);
  clReleaseMemObject(d_scan1g);
  clReleaseMemObject(d_scan_global);
  clReleaseProgram(prog);
  clReleaseCommandQueue(que);
  clReleaseContext(ctx);
  free(vals);
  return 0;
}
