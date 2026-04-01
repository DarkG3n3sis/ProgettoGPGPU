using AMDGPU

function vadd!(c, a, b)
   i = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
   if i ≤ length(a)
       c[i] = a[i] + b[i]
   end
   return
end

a = AMDGPU.ones(Int, 1024)
b = AMDGPU.ones(Int, 1024)
c = AMDGPU.zeros(Int, 1024)

groupsize = 256
gridsize = cld(length(c), groupsize)
@roc groupsize=groupsize gridsize=gridsize vadd!(c, a, b)

@assert (a .+ b) ≈ c