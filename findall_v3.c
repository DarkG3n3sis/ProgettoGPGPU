#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

// Define min macro if not available
#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

void error(const char *err)
{
	fprintf(stderr, "%s\n", err);
	exit(1);
}

cl_event mark_positive(cl_command_queue que, cl_kernel mark_kernel,
	cl_mem d_flags, cl_mem d_input, int nels, size_t lws)
{
	size_t gws = round_mul_up(nels, lws);
	
	printf("mark_positive: %d | %zu = %zu\n", nels, lws, gws);
	cl_int err;
	cl_event ret;

	int arg = 0;
	err = clSetKernelArg(mark_kernel, arg++, sizeof(d_flags), &d_flags);
	ocl_check(err, "set mark_positive arg flags");
	err = clSetKernelArg(mark_kernel, arg++, sizeof(d_input), &d_input);
	ocl_check(err, "set mark_positive arg input");
	err = clSetKernelArg(mark_kernel, arg++, sizeof(nels), &nels);
	ocl_check(err, "set mark_positive arg nels");

	err = clEnqueueNDRangeKernel(que, mark_kernel, 1,
		NULL, &gws, &lws, 0, NULL, &ret);
	ocl_check(err, "enqueue mark_positive");

	return ret;
}

cl_event scan_local(cl_command_queue que, cl_kernel scan_kernel, cl_event wait_for,
	cl_mem d_scan_out, cl_mem d_flags, int nels, size_t lws)
{
	size_t gws = round_mul_up(nels, lws);
	
	printf("scan_local: %d | %zu = %zu\n", nels, lws, gws);
	cl_int err;
	cl_event ret;

	// Verifica limiti memoria locale
	size_t local_mem_size = lws * sizeof(cl_int);
	if (local_mem_size > 48 * 1024) { // Limite tipico 48KB
		fprintf(stderr, "Warning: Local memory size %zu bytes may exceed device limits\n", local_mem_size);
	}

	int arg = 0;
	err = clSetKernelArg(scan_kernel, arg++, sizeof(d_scan_out), &d_scan_out);
	ocl_check(err, "set scan_local arg scan_out");
	err = clSetKernelArg(scan_kernel, arg++, sizeof(d_flags), &d_flags);
	ocl_check(err, "set scan_local arg flags");
	err = clSetKernelArg(scan_kernel, arg++, sizeof(nels), &nels);
	ocl_check(err, "set scan_local arg nels");
	err = clSetKernelArg(scan_kernel, arg++, local_mem_size, NULL);
	ocl_check(err, "set scan_local arg lmem");

	err = clEnqueueNDRangeKernel(que, scan_kernel, 1,
		NULL, &gws, &lws, 1, &wait_for, &ret);
	ocl_check(err, "enqueue scan_local");

	return ret;
}

cl_event extract_sums(cl_command_queue que, cl_kernel extract_kernel, cl_event wait_for,
	cl_mem d_block_sums, cl_mem d_scan_out, int nels, size_t lws)
{
	int num_groups = round_div_up(nels, lws);
	size_t gws = round_mul_up(num_groups, lws); // Fix: use num_groups not num_groups*lws
	
	printf("extract_sums: %d groups | %zu = %zu\n", num_groups, lws, gws);
	cl_int err;
	cl_event ret;

	int arg = 0;
	err = clSetKernelArg(extract_kernel, arg++, sizeof(d_block_sums), &d_block_sums);
	ocl_check(err, "set extract_sums arg block_sums");
	err = clSetKernelArg(extract_kernel, arg++, sizeof(d_scan_out), &d_scan_out);
	ocl_check(err, "set extract_sums arg scan_out");
	err = clSetKernelArg(extract_kernel, arg++, sizeof(nels), &nels);
	ocl_check(err, "set extract_sums arg nels");
	// Rimuovo il 4° argomento per ora - il kernel originale ne ha solo 3

	err = clEnqueueNDRangeKernel(que, extract_kernel, 1,
		NULL, &gws, &lws, 1, &wait_for, &ret);
	ocl_check(err, "enqueue extract_sums");

	return ret;
}

cl_event scan_sums(cl_command_queue que, cl_kernel scan_sums_kernel, cl_event wait_for,
	cl_mem d_block_offsets, cl_mem d_block_sums, int num_blocks, size_t lws)
{
	size_t gws = round_mul_up(num_blocks, lws);
	
	printf("scan_sums: %d blocks | %zu = %zu\n", num_blocks, lws, gws);
	cl_int err;
	cl_event ret;

	// Limita lws per evitare overflow memoria locale
	size_t max_lws = min(lws, 1024); // Limite ragionevole
	size_t local_mem_size = max_lws * sizeof(cl_int);

	int arg = 0;
	err = clSetKernelArg(scan_sums_kernel, arg++, sizeof(d_block_offsets), &d_block_offsets);
	ocl_check(err, "set scan_sums arg block_offsets");
	err = clSetKernelArg(scan_sums_kernel, arg++, sizeof(d_block_sums), &d_block_sums);
	ocl_check(err, "set scan_sums arg block_sums");
	err = clSetKernelArg(scan_sums_kernel, arg++, sizeof(num_blocks), &num_blocks);
	ocl_check(err, "set scan_sums arg num_blocks");
	err = clSetKernelArg(scan_sums_kernel, arg++, local_mem_size, NULL);
	ocl_check(err, "set scan_sums arg lmem");

	err = clEnqueueNDRangeKernel(que, scan_sums_kernel, 1,
		NULL, &gws, &max_lws, 1, &wait_for, &ret);
	ocl_check(err, "enqueue scan_sums");

	return ret;
}

cl_event apply_offsets(cl_command_queue que, cl_kernel apply_kernel, cl_event wait_for,
	cl_mem d_scan_out, cl_mem d_block_offsets, int nels, size_t lws)
{
	size_t gws = round_mul_up(nels, lws);
	
	printf("apply_offsets: %d | %zu = %zu\n", nels, lws, gws);
	cl_int err;
	cl_event ret;

	int arg = 0;
	err = clSetKernelArg(apply_kernel, arg++, sizeof(d_scan_out), &d_scan_out);
	ocl_check(err, "set apply_offsets arg scan_out");
	err = clSetKernelArg(apply_kernel, arg++, sizeof(d_block_offsets), &d_block_offsets);
	ocl_check(err, "set apply_offsets arg block_offsets");
	err = clSetKernelArg(apply_kernel, arg++, sizeof(nels), &nels);
	ocl_check(err, "set apply_offsets arg nels");

	err = clEnqueueNDRangeKernel(que, apply_kernel, 1,
		NULL, &gws, &lws, 1, &wait_for, &ret);
	ocl_check(err, "enqueue apply_offsets");

	return ret;
}

cl_event findall_final(cl_command_queue que, cl_kernel findall_kernel, cl_event wait_for,
	cl_mem d_output, cl_mem d_flags, cl_mem d_scan_out, int nels, size_t lws)
{
	size_t gws = round_mul_up(nels, lws);
	
	printf("findall_final: %d | %zu = %zu\n", nels, lws, gws);
	cl_int err;
	cl_event ret;

	int arg = 0;
	err = clSetKernelArg(findall_kernel, arg++, sizeof(d_output), &d_output);
	ocl_check(err, "set findall_final arg output");
	err = clSetKernelArg(findall_kernel, arg++, sizeof(d_flags), &d_flags);
	ocl_check(err, "set findall_final arg flags");
	err = clSetKernelArg(findall_kernel, arg++, sizeof(d_scan_out), &d_scan_out);
	ocl_check(err, "set findall_final arg scan_out");
	err = clSetKernelArg(findall_kernel, arg++, sizeof(nels), &nels);
	ocl_check(err, "set findall_final arg nels");

	err = clEnqueueNDRangeKernel(que, findall_kernel, 1,
		NULL, &gws, &lws, 1, &wait_for, &ret);
	ocl_check(err, "enqueue findall_final");

	return ret;
}

void verify_result(const cl_int *output, int count, const char *input, int nels)
{
	printf("Found %d positive elements", count);
	
	// Verifica overflow del contatore
	if (count < 0) {
		fprintf(stderr, "\nERROR: Count is negative (%d) - integer overflow detected!\n", count);
		return;
	}
	
	// Show only a sample of indices if there are too many
	if (count <= 20 && count > 0) {
		printf(" at indices: ");
		for (int i = 0; i < count; i++) {
			printf("%d ", output[i]);
		}
	} else if (count > 0) {
		printf(" at indices (first 10): ");
		int show_count = min(10, count);
		for (int i = 0; i < show_count; i++) {
			printf("%d ", output[i]);
		}
		if (count > 10) {
			printf("... (and %d more)", count - 10);
		}
	}
	printf("\n");
	
	// Verify correctness
	long long expected_count = 0; // Use long long to detect overflow
	for (int i = 0; i < nels; i++) {
		if (input[i] > 0) expected_count++;
	}
	
	// Check for overflow in expected count
	if (expected_count > INT_MAX) {
		fprintf(stderr, "Expected count %lld exceeds INT_MAX (%d) - array too large!\n", 
			expected_count, INT_MAX);
		return;
	}
	
	if ((int)expected_count != count) {
		fprintf(stderr, "Count mismatch: expected %lld, got %d\n", expected_count, count);
	} else {
		printf("Verification: PASSED ✓\n");
	}
}

int main(int argc, char *argv[])
{
	if (argc < 3) error("usage: findall <lws> <array_size>");

	int lws = atoi(argv[1]);
	if (lws <= 0) error("lws must be positive");
	if (lws > 1024) {
		fprintf(stderr, "Warning: lws %d is very large, reducing to 1024\n", lws);
		lws = 1024;
	}
	
	long long nels_ll = atoll(argv[2]); // Use long long to check range
	if (nels_ll <= 0) error("array_size must be positive");
	if (nels_ll > INT_MAX) error("array_size too large (max 2^31-1)");
	
	int nels = (int)nels_ll;
	
	// Check for reasonable memory limits
	size_t total_memory = (size_t)nels * (sizeof(char) * 2 + sizeof(cl_int) * 2);
	if (total_memory > 4ULL * 1024 * 1024 * 1024) { // 4GB limit
		fprintf(stderr, "Warning: Total memory usage %.2f GB may be too large\n", 
			total_memory / (1024.0 * 1024.0 * 1024.0));
	}

	// Test data with fixed seed for reproducibility
	srand(42); // Fixed seed for consistent results
	
	char *vals = malloc(nels * sizeof(char));
	if (!vals) error("failed to allocate memory for test array");
	
	for (int i = 0; i < nels; i++) {
		vals[i] = (char)(rand() % 5 - 2); // Generate values from -2 to 2
	}
	
	printf("Array size: %d elements\n", nels);
	if (nels <= 20) {
		printf("Input array: ");
		for (int i = 0; i < nels; i++) {
			printf("%d ", vals[i]);
		}
		printf("\n");
	}

	size_t input_memsize = nels * sizeof(char);
	size_t flags_memsize = nels * sizeof(char);
	size_t scan_memsize = nels * sizeof(cl_int);
	
	int num_groups = round_div_up(nels, lws);
	size_t sums_memsize = num_groups * sizeof(cl_int);

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("findall.ocl", ctx, d);

	cl_int err;
	
	// Create kernels
	cl_kernel mark_kernel = clCreateKernel(prog, "mark_positive_kernel", &err);
	ocl_check(err, "create mark_positive_kernel");
	cl_kernel scan_kernel = clCreateKernel(prog, "scan_local_kernel", &err);
	ocl_check(err, "create scan_local_kernel");
	cl_kernel extract_kernel = clCreateKernel(prog, "extract_block_sums", &err);
	ocl_check(err, "create extract_block_sums");
	cl_kernel scan_sums_kernel = clCreateKernel(prog, "scan_block_sums_kernel", &err);
	ocl_check(err, "create scan_block_sums_kernel");
	cl_kernel apply_kernel = clCreateKernel(prog, "apply_block_offsets", &err);
	ocl_check(err, "create apply_block_offsets");
	cl_kernel findall_kernel = clCreateKernel(prog, "findall_gpu", &err);
	ocl_check(err, "create findall_gpu");

	// Create buffers
	cl_mem d_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		input_memsize, vals, &err);
	ocl_check(err, "create d_input");
	cl_mem d_flags = clCreateBuffer(ctx, CL_MEM_READ_WRITE, flags_memsize, NULL, &err);
	ocl_check(err, "create d_flags");
	cl_mem d_scan_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, scan_memsize, NULL, &err);
	ocl_check(err, "create d_scan_out");
	cl_mem d_block_sums = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sums_memsize, NULL, &err);
	ocl_check(err, "create d_block_sums");
	cl_mem d_block_offsets = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sums_memsize, NULL, &err);
	ocl_check(err, "create d_block_offsets");

	printf("Launching mark_positive...\n");
	cl_event mark_evt = mark_positive(que, mark_kernel, d_flags, d_input, nels, lws);
	err = clWaitForEvents(1, &mark_evt);
	ocl_check(err, "wait for mark_positive");
	printf("mark_positive completed\n");

	printf("Launching scan_local...\n");
	cl_event scan_evt = scan_local(que, scan_kernel, mark_evt, d_scan_out, d_flags, nels, lws);
	err = clWaitForEvents(1, &scan_evt);
	ocl_check(err, "wait for scan_local");
	printf("scan_local completed\n");

	printf("Launching extract_sums...\n");
	cl_event extract_evt = extract_sums(que, extract_kernel, scan_evt, d_block_sums, d_scan_out, nels, lws);
	err = clWaitForEvents(1, &extract_evt);
	ocl_check(err, "wait for extract_sums");
	printf("extract_sums completed\n");

	printf("Launching scan_sums...\n");
	cl_event scan_sums_evt = scan_sums(que, scan_sums_kernel, extract_evt, d_block_offsets, d_block_sums, num_groups, lws);
	err = clWaitForEvents(1, &scan_sums_evt);
	ocl_check(err, "wait for scan_sums");
	printf("scan_sums completed\n");

	printf("Launching apply_offsets...\n");
	cl_event apply_evt = apply_offsets(que, apply_kernel, scan_sums_evt, d_scan_out, d_block_offsets, nels, lws);
	err = clWaitForEvents(1, &apply_evt);
	ocl_check(err, "wait for apply_offsets");
	printf("apply_offsets completed\n");

	// Get count
	printf("Mapping scan buffer to read count...\n");
	cl_event map_scan_evt;
	cl_int *h_scan = clEnqueueMapBuffer(que, d_scan_out, CL_TRUE, CL_MAP_READ, 0, scan_memsize,
		1, &apply_evt, &map_scan_evt, &err);
	ocl_check(err, "map scan_out");

	int count = h_scan[nels - 1];
	printf("Total positive elements: %d\n", count);

	clEnqueueUnmapMemObject(que, d_scan_out, h_scan, 0, NULL, NULL);

	if (count > 0) {
		// Verifica che count sia ragionevole
		if (count > nels) {
			fprintf(stderr, "ERROR: Count %d exceeds array size %d - data corruption!\n", count, nels);
			goto cleanup;
		}

		// Create output buffer and get final result
		size_t output_size = count * sizeof(cl_int);
		cl_mem d_output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
			output_size, NULL, &err);
		ocl_check(err, "create d_output");

		printf("Launching findall_final...\n");
		cl_event findall_evt = findall_final(que, findall_kernel, apply_evt, d_output, d_flags, d_scan_out, nels, lws);
		err = clWaitForEvents(1, &findall_evt);
		ocl_check(err, "wait for findall_final");
		printf("findall_final completed\n");

		printf("Mapping output buffer...\n");
		cl_event map_output_evt;
		cl_int *h_output = clEnqueueMapBuffer(que, d_output, CL_TRUE, CL_MAP_READ, 0, output_size,
			1, &findall_evt, &map_output_evt, &err);
		ocl_check(err, "map output");
		printf("Output buffer mapped successfully\n");

		verify_result(h_output, count, vals, nels);

		// Performance reporting with bandwidth analysis
		double mark_time = runtime_ms(mark_evt);
		double scan_time = runtime_ms(scan_evt);
		double extract_time = runtime_ms(extract_evt);
		double scan_sums_time = runtime_ms(scan_sums_evt);
		double apply_time = runtime_ms(apply_evt);
		double findall_time = runtime_ms(findall_evt);
		double total_time = total_runtime_ms(mark_evt, findall_evt);

		// Memory sizes for bandwidth calculations
		size_t input_bytes = nels * sizeof(char);
		size_t flags_bytes = nels * sizeof(char);
		size_t scan_bytes = nels * sizeof(cl_int);
		size_t sums_bytes = num_groups * sizeof(cl_int);
		size_t output_bytes = count * sizeof(cl_int);

		printf("Performance Analysis:\n");
		printf("=====================\n");

		// mark_positive: read input, write flags
		double mark_bandwidth = (input_bytes + flags_bytes) / mark_time / 1.0e6;
		printf("mark_positive: %.3f ms, %.2f GE/s, %.2f GB/s\n", 
			mark_time, nels/mark_time/1.0e6, mark_bandwidth);

		// scan_local: read flags, write scan_out 
		double scan_bandwidth = (flags_bytes + scan_bytes) / scan_time / 1.0e6;
		printf("scan_local: %.3f ms, %.2f GE/s, %.2f GB/s\n",
			scan_time, nels/scan_time/1.0e6, scan_bandwidth);

		// extract_sums: read scan_out, write block_sums
		double extract_bandwidth = (scan_bytes + sums_bytes) / extract_time / 1.0e6;
		printf("extract_sums: %.3f ms, %.2f GE/s, %.2f GB/s\n",
			extract_time, num_groups/extract_time/1.0e6, extract_bandwidth);

		// scan_sums: read+write block_sums, write block_offsets  
		double scan_sums_bandwidth = (2*sums_bytes) / scan_sums_time / 1.0e6;
		printf("scan_sums: %.3f ms, %.2f GE/s, %.2f GB/s\n",
			scan_sums_time, num_groups/scan_sums_time/1.0e6, scan_sums_bandwidth);

		// apply_offsets: read+write scan_out, read block_offsets
		double apply_bandwidth = (2*scan_bytes + sums_bytes) / apply_time / 1.0e6;
		printf("apply_offsets: %.3f ms, %.2f GE/s, %.2f GB/s\n",
			apply_time, nels/apply_time/1.0e6, apply_bandwidth);

		// findall_final: read flags+scan_out, write output
		double findall_bandwidth = (flags_bytes + scan_bytes + output_bytes) / findall_time / 1.0e6;
		printf("findall_final: %.3f ms, %.2f GE/s, %.2f GB/s\n",
			findall_time, nels/findall_time/1.0e6, findall_bandwidth);

		printf("---------------------\n");
		printf("TOTAL: %.3f ms, %.2f GE/s\n", total_time, nels/total_time/1.0e6);

		// Overall pipeline bandwidth
		double total_bytes = input_bytes + 3*flags_bytes + 4*scan_bytes + 4*sums_bytes + output_bytes;
		printf("Pipeline bandwidth: %.2f GB/s\n", total_bytes/total_time/1.0e6);

		// Efficiency vs theoretical peak
		printf("\nMemory Analysis:\n");
		printf("Input: %.2f MB, Flags: %.2f MB, Scan: %.2f MB\n",
			input_bytes/1.0e6, flags_bytes/1.0e6, scan_bytes/1.0e6);
		printf("Sums: %.2f MB, Output: %.2f MB\n", 
			sums_bytes/1.0e6, output_bytes/1.0e6);
		printf("Total allocated: %.2f MB\n", 
			(input_bytes + flags_bytes + scan_bytes + 2*sums_bytes + output_bytes)/1.0e6);

		clEnqueueUnmapMemObject(que, d_output, h_output, 0, NULL, NULL);
		clReleaseMemObject(d_output);
	}

cleanup:
	err = clFinish(que);
	ocl_check(err, "finish");

	// Cleanup
	clReleaseKernel(mark_kernel);
	clReleaseKernel(scan_kernel);
	clReleaseKernel(extract_kernel);
	clReleaseKernel(scan_sums_kernel);
	clReleaseKernel(apply_kernel);
	clReleaseKernel(findall_kernel);
	clReleaseMemObject(d_input);
	clReleaseMemObject(d_flags);
	clReleaseMemObject(d_scan_out);
	clReleaseMemObject(d_block_sums);
	clReleaseMemObject(d_block_offsets);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);
	free(vals);

	return 0;
}