using AMDGPU, KernelAbstractions, Test, Random

@kernel function mul2_kernel(A)
  I = @index(Global)
  A[I] = 2 * A[I]
end

function mymul(A)
    A .= 1.0
    backend = get_backend(A)
    ev = mul2_kernel(backend, 64)(A, ndrange=size(A))
    synchronize(backend)
    all(A .== 2.0)
end

A = ROCArray(ones(1024, 1024))
backend = get_backend(A)
mul2_kernel(backend, 64)(A, ndrange=size(A))
synchronize(backend)
print(backend)
print(all(A .== 2.0))

