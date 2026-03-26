using AMDGPU


using AMDGPU

a = AMDGPU.fill(1.0f0, 1024)
b = AMDGPU.fill(2.0f0, 1024)

c = a .+ b  # Vettorizzato, eseguito sulla GPU se gli array di partenza sono ROCArray

print(typeof(c))