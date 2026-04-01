using OpenCL

platform = cl.platforms()[2]  # seleziona la seconda piattaforma
device = cl.devices(platform)[1]  # ad esempio il primo device su quella piattaforma
cl.platform!(platform)

dims = (2,)
a = round.(rand(Float32, dims) * 1)
b = round.(rand(Float32, dims) * 1)
c = similar(a)

d_a = CLArray(a)
d_b = CLArray(b)
d_c = CLArray(c)

function vadd(a, b, c)
    i = get_global_id()
    @inbounds c[i] = a[i] + b[i]
    return
end

len = prod(dims)
@opencl global_size=len vadd(d_a, d_b, d_c)