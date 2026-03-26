/* findall_lmem.c
 *
 * Scan per blocchi con local memory — ottimizzata per GPU reale.
 *
 * Pipeline (4 kernel, O(N) traffico memoria):
 *   1. mark_and_scan_local : mark + prefix-sum per blocco in lmem
 *   2. scan_block_sums     : prefix-sum esclusiva sulle somme dei blocchi
 *   3. apply_offsets       : aggiunge offset blocco -> scan globale
 *   4. findall_gpu         : scatter indici positivi
 *
 * Limite: num_groups <= lws  =>  nels <= lws^2
 *   lws=256 -> max nels ~65K
 *   lws=512 -> max nels ~262K
 *   lws=1024 -> max nels ~1M
 * Per array piu' grandi usa findall_final (versione ping-pong globale).
 */

#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

void error(const char *err) { fprintf(stderr, "%s\n", err); exit(1); }

/* --- wrappers ------------------------------------------------------------ */

cl_event run_mark_and_scan_local(
    cl_command_queue que, cl_kernel k,
    cl_mem d_flags, cl_mem d_scan_local, cl_mem d_block_sums,
    cl_mem d_input, int nels, size_t lws)
{
    size_t gws = round_mul_up(nels, lws);
    cl_int err; cl_event ret; int arg = 0;

    err = clSetKernelArg(k, arg++, sizeof(d_flags),      &d_flags);      ocl_check(err, "mark arg flags");
    err = clSetKernelArg(k, arg++, sizeof(d_scan_local), &d_scan_local); ocl_check(err, "mark arg scan_local");
    err = clSetKernelArg(k, arg++, sizeof(d_block_sums), &d_block_sums); ocl_check(err, "mark arg block_sums");
    err = clSetKernelArg(k, arg++, sizeof(d_input),      &d_input);      ocl_check(err, "mark arg input");
    err = clSetKernelArg(k, arg++, sizeof(nels),         &nels);         ocl_check(err, "mark arg nels");
    err = clSetKernelArg(k, arg++, lws * sizeof(cl_int), NULL);          ocl_check(err, "mark arg lmem");

    err = clEnqueueNDRangeKernel(que, k, 1, NULL, &gws, &lws, 0, NULL, &ret);
    ocl_check(err, "enqueue mark_and_scan_local");
    return ret;
}

cl_event run_scan_block_sums(
    cl_command_queue que, cl_kernel k, cl_event wait_for,
    cl_mem d_block_offsets, cl_mem d_block_sums, int num_blocks, size_t lws)
{
    size_t gws = round_mul_up(num_blocks, lws);
    cl_int err; cl_event ret; int arg = 0;

    err = clSetKernelArg(k, arg++, sizeof(d_block_offsets), &d_block_offsets); ocl_check(err, "sbs arg offsets");
    err = clSetKernelArg(k, arg++, sizeof(d_block_sums),    &d_block_sums);    ocl_check(err, "sbs arg sums");
    err = clSetKernelArg(k, arg++, sizeof(num_blocks),      &num_blocks);      ocl_check(err, "sbs arg num_blocks");
    err = clSetKernelArg(k, arg++, lws * sizeof(cl_int),    NULL);             ocl_check(err, "sbs arg lmem");

    err = clEnqueueNDRangeKernel(que, k, 1, NULL, &gws, &lws, 1, &wait_for, &ret);
    ocl_check(err, "enqueue scan_block_sums");
    return ret;
}

cl_event run_apply_offsets(
    cl_command_queue que, cl_kernel k, cl_event wait_for,
    cl_mem d_scan_global, cl_mem d_scan_local, cl_mem d_block_offsets,
    int nels, size_t lws)
{
    size_t gws = round_mul_up(nels, lws);
    cl_int err; cl_event ret; int arg = 0;

    err = clSetKernelArg(k, arg++, sizeof(d_scan_global),   &d_scan_global);   ocl_check(err, "apply arg scan_global");
    err = clSetKernelArg(k, arg++, sizeof(d_scan_local),    &d_scan_local);    ocl_check(err, "apply arg scan_local");
    err = clSetKernelArg(k, arg++, sizeof(d_block_offsets), &d_block_offsets); ocl_check(err, "apply arg block_offsets");
    err = clSetKernelArg(k, arg++, sizeof(nels),            &nels);            ocl_check(err, "apply arg nels");

    err = clEnqueueNDRangeKernel(que, k, 1, NULL, &gws, &lws, 1, &wait_for, &ret);
    ocl_check(err, "enqueue apply_offsets");
    return ret;
}

cl_event run_findall(
    cl_command_queue que, cl_kernel k, cl_event wait_for,
    cl_mem d_output, cl_mem d_flags, cl_mem d_scan_global,
    int nels, size_t lws)
{
    size_t gws = round_mul_up(nels, lws);
    cl_int err; cl_event ret; int arg = 0;

    err = clSetKernelArg(k, arg++, sizeof(d_output),      &d_output);      ocl_check(err, "find arg output");
    err = clSetKernelArg(k, arg++, sizeof(d_flags),       &d_flags);       ocl_check(err, "find arg flags");
    err = clSetKernelArg(k, arg++, sizeof(d_scan_global), &d_scan_global); ocl_check(err, "find arg scan_global");
    err = clSetKernelArg(k, arg++, sizeof(nels),          &nels);          ocl_check(err, "find arg nels");

    err = clEnqueueNDRangeKernel(que, k, 1, NULL, &gws, &lws, 1, &wait_for, &ret);
    ocl_check(err, "enqueue findall_gpu");
    return ret;
}

/* --- verifica ------------------------------------------------------------ */

void verify_result(const cl_int *output, int count, const char *input, int nels)
{
    printf("Found %d positive elements", count);
    if (count > 0 && count <= 20) {
        printf(" at indices:");
        for (int i = 0; i < count; i++) printf(" %d", output[i]);
    } else if (count > 0) {
        printf(" at indices (first 10):");
        for (int i = 0; i < 10; i++) printf(" %d", output[i]);
        printf(" ... (%d more)", count - 10);
    }
    printf("\n");

    int expected = 0;
    for (int i = 0; i < nels; i++) if (input[i] > 0) expected++;
    if (expected != count)
        fprintf(stderr, "FAILED: expected %d, got %d\n", expected, count);
    else
        printf("Verification: PASSED\n");
}

/* --- main ---------------------------------------------------------------- */

int main(int argc, char *argv[])
{
    if (argc < 3) error("usage: findall_lmem <lws> <nels>");

    int lws_i = atoi(argv[1]);
    if (lws_i <= 0) error("lws must be positive");
    if (lws_i > 1024) { lws_i = 1024; fprintf(stderr, "lws clamped to 1024\n"); }
    size_t lws = (size_t)lws_i;

    long long nels_ll = atoll(argv[2]);
    if (nels_ll <= 0) error("nels must be positive");
    if (nels_ll > INT_MAX) error("nels too large");
    int nels = (int)nels_ll;

    int num_groups = round_div_up(nels, lws_i);
    if (num_groups > lws_i) {
        fprintf(stderr,
            "ERROR: num_groups=%d > lws=%d (max nels con lws=%d e' %d).\n"
            "       Aumenta lws oppure usa findall_final per array piu' grandi.\n",
            num_groups, lws_i, lws_i, lws_i * lws_i);
        exit(1);
    }

    srand(42);
    char *vals = malloc(nels * sizeof(char));
    if (!vals) error("malloc failed");
    for (int i = 0; i < nels; i++)
        vals[i] = (char)(rand() % 5 - 2);

    printf("Array size: %d  lws: %zu  num_groups: %d\n", nels, lws, num_groups);
    if (nels <= 20) {
        printf("Input:");
        for (int i = 0; i < nels; i++) printf(" %d", vals[i]);
        printf("\n");
    }

    /* OpenCL setup */
    cl_platform_id p   = select_platform();
    cl_device_id   d   = select_device(p);
    cl_context     ctx = create_context(p, d);
    cl_command_queue que = create_queue(ctx, d);
    cl_program     prog = create_program("findall_lmem.ocl", ctx, d);
    cl_int err;

    cl_kernel k_mark  = clCreateKernel(prog, "mark_and_scan_local", &err);
    ocl_check(err, "create mark_and_scan_local");
    cl_kernel k_scan  = clCreateKernel(prog, "scan_block_sums", &err);
    ocl_check(err, "create scan_block_sums");
    cl_kernel k_apply = clCreateKernel(prog, "apply_offsets", &err);
    ocl_check(err, "create apply_offsets");
    cl_kernel k_find  = clCreateKernel(prog, "findall_gpu", &err);
    ocl_check(err, "create findall_gpu");

    size_t scan_sz = nels       * sizeof(cl_int);
    size_t sums_sz = num_groups * sizeof(cl_int);

    cl_mem d_input          = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                  nels*sizeof(char), vals, &err); ocl_check(err, "d_input");
    cl_mem d_flags          = clCreateBuffer(ctx, CL_MEM_READ_WRITE, scan_sz, NULL, &err); ocl_check(err, "d_flags");
    cl_mem d_scan_local     = clCreateBuffer(ctx, CL_MEM_READ_WRITE, scan_sz, NULL, &err); ocl_check(err, "d_scan_local");
    cl_mem d_scan_global    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, scan_sz, NULL, &err); ocl_check(err, "d_scan_global");
    cl_mem d_block_sums     = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sums_sz, NULL, &err); ocl_check(err, "d_block_sums");
    cl_mem d_block_offsets  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sums_sz, NULL, &err); ocl_check(err, "d_block_offsets");

    /* ------------------------------------------------------------------ */
    /* Pipeline                                                             */
    /* ------------------------------------------------------------------ */
    cl_event e_mark  = run_mark_and_scan_local(que, k_mark,
                           d_flags, d_scan_local, d_block_sums,
                           d_input, nels, lws);

    cl_event e_scan  = run_scan_block_sums(que, k_scan, e_mark,
                           d_block_offsets, d_block_sums, num_groups, lws);

    cl_event e_apply = run_apply_offsets(que, k_apply, e_scan,
                           d_scan_global, d_scan_local, d_block_offsets,
                           nels, lws);

    /* Leggi count dall'ultimo elemento della scan globale */
    cl_event e_map;
    cl_int *h_scan = clEnqueueMapBuffer(que, d_scan_global, CL_TRUE, CL_MAP_READ,
        0, scan_sz, 1, &e_apply, &e_map, &err);
    ocl_check(err, "map d_scan_global");

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
        cl_mem d_output = clCreateBuffer(ctx,
            CL_MEM_WRITE_ONLY|CL_MEM_ALLOC_HOST_PTR, out_sz, NULL, &err);
        ocl_check(err, "d_output");

        cl_event e_find = run_findall(que, k_find, e_apply,
                              d_output, d_flags, d_scan_global, nels, lws);

        err = clWaitForEvents(1, &e_find);
        ocl_check(err, "wait e_find");

        cl_event e_map_out;
        cl_int *h_out = clEnqueueMapBuffer(que, d_output, CL_TRUE, CL_MAP_READ,
            0, out_sz, 1, &e_find, &e_map_out, &err);
        ocl_check(err, "map d_output");

        verify_result(h_out, count, vals, nels);

        /* Performance report */
        double ms_mark  = runtime_ms(e_mark);
        double ms_scan  = runtime_ms(e_scan);
        double ms_apply = runtime_ms(e_apply);
        double ms_find  = runtime_ms(e_find);
        double ms_total = total_runtime_ms(e_mark, e_find);

        size_t in_b   = nels       * sizeof(char);
        size_t sc_b   = nels       * sizeof(cl_int);
        size_t sm_b   = num_groups * sizeof(cl_int);
        size_t out_b  = count      * sizeof(cl_int);

        printf("\nPerformance:\n");
        printf("  mark+scan_local:  %7.3f ms  bw: %.2f GB/s\n", ms_mark,
            (in_b + 2*sc_b + sm_b) / ms_mark / 1.0e6);
        printf("  scan_block_sums:  %7.3f ms  bw: %.2f GB/s\n", ms_scan,
            (2.0*sm_b) / ms_scan / 1.0e6);
        printf("  apply_offsets:    %7.3f ms  bw: %.2f GB/s\n", ms_apply,
            (2.0*sc_b + sm_b) / ms_apply / 1.0e6);
        printf("  findall_scatter:  %7.3f ms  bw: %.2f GB/s\n", ms_find,
            (2.0*sc_b + out_b) / ms_find / 1.0e6);
        printf("  TOTAL:            %7.3f ms  throughput: %.2f GE/s\n",
            ms_total, nels / ms_total / 1.0e6);

        clEnqueueUnmapMemObject(que, d_output, h_out, 0, NULL, NULL);
        clReleaseEvent(e_map_out);
        clReleaseEvent(e_find);
        clReleaseMemObject(d_output);
    }

cleanup:
    err = clFinish(que);
    ocl_check(err, "finish");

    clReleaseEvent(e_mark);
    clReleaseEvent(e_scan);
    clReleaseEvent(e_apply);

    clReleaseKernel(k_mark);
    clReleaseKernel(k_scan);
    clReleaseKernel(k_apply);
    clReleaseKernel(k_find);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_flags);
    clReleaseMemObject(d_scan_local);
    clReleaseMemObject(d_scan_global);
    clReleaseMemObject(d_block_sums);
    clReleaseMemObject(d_block_offsets);
    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);
    free(vals);
    return 0;
}
