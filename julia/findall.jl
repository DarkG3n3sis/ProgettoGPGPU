#ENV["JULIA_DEBUG"] = "AMDGPU"
using KernelAbstractions
using AMDGPU
using Cthulhu


if false
    const KADevice = ROCBackend()
else
    const KADevice = CPU()
end

const GROUP_SIZE = 16 #lws


#Kernel 1: Costruisce flags binari
@kernel function mark_positive_kernel!(input, flags, N)
    i = @index(Global)
    if i <= N
        flags[i] = input[i] > 0 ? 1 : 0
        #@print("Thread ", i, ": input = ", input[i], ", flag = ", flags[i], "\n")
    end
end


#Kernel 2: Scan parziale in blocco
@kernel function scan_local_kernel!(flags, scan_out, N, group_sz, niterations)
    i = @index(Global)
    lidx = @index(Local)
    #@private group_sz = Int32(only(@groupsize()))
    #group_sz = group_sz[1]


    if i == 0
        @print("group_sz = ", group_sz, "\n")
        @print("@groupsize() = ", @groupsize(), "\n")
        @print("typeof(@groupsize()) = ", typeof(group_sz), "\n")
        @print("@groupsize()[1] = ", @groupsize()[1], "s\n")
        @print("typeof(@groupsize()[1]) = ", typeof(@groupsize()[1]), "\n")
    end
    #@print("Type of @groupsize(): ", typeof(group_tuple), "\n")
    #@print("Value of @groupsize(): ", group_tuple, "\n")
    #@print("Is tuple? ", isa(group_tuple, Tuple), "\n")



    temp = @localmem(Int8, group_sz)

    if i <= N
        temp[lidx] = flags[i]
    else
        temp[lidx] = 0
    end

    @synchronize()

    #Implementazioni simili
    if true
        for d in Int32(1):niterations
            stride = Int32(1) << (d - 1)
            @synchronize()
            if lidx > stride
                temp[lidx] += temp[lidx - stride]
            end
        end
    else # da problemi con isless stride < niterations
        @private stride = Int32(1)
        while stride < niterations
            @synchronize()
            if lidx > stride
                temp[lidx] += temp[lidx - stride]
            end
            stride *= 2
        end
    end
 
    @synchronize()
    if i <= N
        scan_out[i] = temp[lidx]
    end
    
    if i == N @print("FInito: scan_local_kernel!", "\n") end
end


#Funzione alternativa non da problemi senza cilci, non ottimale ne scalabile
@kernel function _scan_local_kernel!(flags, scan_out, N)
    i = @index(Global)
    lidx = @index(Local)
    
    temp = @localmem(Int32, GROUP_SIZE)
    
    if i <= N
        temp[lidx] = Int32(flags[i])
    else
        temp[lidx] = Int32(0)
    end
    
    @synchronize()
    

    @synchronize()
    if lidx > Int32(1)
        temp[lidx] += temp[lidx - Int32(1)]
    end
    
    @synchronize()
    if lidx > Int32(2)
        temp[lidx] += temp[lidx - Int32(2)]
    end
    
    @synchronize()
    if lidx > Int32(4)
        temp[lidx] += temp[lidx - Int32(4)]
    end
    
    @synchronize()
    
    if i <= N
        scan_out[i] = temp[lidx]
    end
end


#Kernel 3: Estrai somma per blocco
@kernel function extract_block_sums!(scan_out, block_sums, N, group_sz)
    gidx = @index(Group)
    lidx = @index(Local)
    #group_sz = @groupsize()

    last_thread = (gidx - 1) * group_sz + group_sz

    if lidx == group_sz
        block_sums[gidx] = (last_thread <= N) ? scan_out[last_thread] : scan_out[N]
    end
end


#Kernel 4: Scan degli offset
@kernel function scan_block_sums_kernel!(block_sums, block_offsets, num_blocks, group_sz, niterations)
    lidx = @index(Local)
    i = lidx
    #group_sz = @groupsize()

    temp = @localmem(Int32, group_sz)

    if i <= num_blocks
        temp[lidx] = block_sums[i]
    else
        temp[lidx] = 0
    end

    @private stride = Int32(0)

    @synchronize()
    
    if true
        for d in Int32(1):niterations
            stride = Int32(Int32(1) << (d - 1))
            @synchronize()
            if lidx > stride
                temp[lidx] += temp[lidx - stride]
            end
        end
    else
        @private stride = Int32(1)
        while stride < niterations
            @synchronize()
            if lidx > stride
                temp[lidx] += temp[lidx - stride]
            end
            stride *= 2
        end
    end
    
    @synchronize()

    if i <= num_blocks
        block_offsets[i] = (i == 1) ? 0 : temp[i - 1]
    end
end


#Kernel 5: Aggiunge offset per blocco
@kernel function apply_block_offsets!(scan_out, block_offsets, N)
    i = @index(Global)
    group_sz = @groupsize()
    group_sz = group_sz[1]
    gidx = (i - 1) ÷ group_sz + 1

    if i <= N
        scan_out[i] += block_offsets[gidx]
    end
end


#Kernel 6: Costruisce output finale
@kernel function findall_gpu(flags, scan_out, output_indices, N)
    i = @index(Global)
    if i <= N && flags[i] == 1
        out_idx = scan_out[i]
        output_indices[out_idx] = i
    end
end


#Funzione wrapper vecchia
function _gpu_findall_positive!(input::ROCArray{Int8})
    N = length(input)
    num_blocks = cld(N, GROUP_SIZE)
    global_size = cld(N, GROUP_SIZE) * GROUP_SIZE

    flags = AMDGPU.zeros(Int8, N)
    scan_out = AMDGPU.zeros(Int32, N)
    block_sums = AMDGPU.zeros(Int32, num_blocks)
    block_offsets = AMDGPU.zeros(Int32, num_blocks)

    mark_positive_kernel!(KADevice, N, GROUP_SIZE)(input, flags, N)
    
    #err = try
    scan_local_kernel!(KADevice, global_size, GROUP_SIZE)(flags, scan_out, N)
    #catch e
    #e
    #end
    #code_typed(err; interactive = true)
    if false
    extract_block_sums!(KADevice, num_blocks, GROUP_SIZE)(scan_out, block_sums, N)

    scan_block_sums_kernel!(KADevice, num_blocks, num_blocks)(block_sums, block_offsets, num_blocks)
    
    apply_block_offsets!(KADevice, N, GROUP_SIZE)(scan_out, block_offsets, N)
    
    total_found = AMDGPU.zeros(Int32, 1)
    KernelAbstractions.copyto!(KADevice, total_found, scan_out[end:end])
    wait(total_found)
    count = Int32(total_found[1])
    output_indices = AMDGPU.zeros(Int32, count)
    #count = Array(scan_out[end:end])[1]
    #output_indices = AMDGPU.zeros(Int32, count)
    #print(count)

    findall_gpu(KADevice, N, GROUP_SIZE)(flags, scan_out, output_indices, N)
    else
        return flags
    end
    return output_indices
end

#Funzione wrapper
function gpu_findall_positive!(input)
    #print(typeof(input))
    N = length(input)
    num_blocks = cld(N, GROUP_SIZE) #Numero di WorkGroups
    niterations = Int32(log2(GROUP_SIZE)) #Numero di iterazioni per la riduzione

    flags = KernelAbstractions.zeros(KADevice, Int8, N)
    scan_out = KernelAbstractions.zeros(KADevice, Int32, N)
    block_sums = KernelAbstractions.zeros(KADevice, Int32, num_blocks)
    block_offsets = KernelAbstractions.zeros(KADevice, Int32, num_blocks)

    #Costruisce flags
    global_size = cld(N, GROUP_SIZE) * GROUP_SIZE
    mark_positive_kernel!(KADevice, global_size, GROUP_SIZE)(input, flags, N)
    KernelAbstractions.synchronize(KADevice)

    #Scan per blocco
    scan_local_kernel!(KADevice, global_size, GROUP_SIZE)(flags, scan_out, N, GROUP_SIZE, niterations)
    KernelAbstractions.synchronize(KADevice)
    if false
    
    #Estrae somma per blocco
    global_size_blocks = num_blocks * GROUP_SIZE
    extract_block_sums!(KADevice, global_size_blocks, GROUP_SIZE)(scan_out, block_sums, N, GROUP_SIZE)
    
    #Scan dei block_sums
    global_size_sums = cld(num_blocks, GROUP_SIZE) * GROUP_SIZE
    scan_block_sums_kernel!(KADevice, global_size_sums, GROUP_SIZE)(block_sums, block_offsets, num_blocks, GROUP_SIZE, niterations)
    

    #Aggiunge offset globale
    apply_block_offsets!(KADevice, global_size, GROUP_SIZE)(scan_out, block_offsets, N)


    #Alloca e riempie output
    total_found = KernelAbstractions.zeros(KADevice, Int32, 1)
    KernelAbstractions.copyto!(KADevice, total_found, scan_out[N:N])  # scan_out[end:end]
    wait(total_found)

    count = Int32(Array(total_found)[1])
    output_indices = KernelAbstractions.zeros(Int32, count)

    findall_gpu(KADevice, global_size, GROUP_SIZE)(flags, scan_out, output_indices, N)
    else
        return scan_out
    end
    return output_indices
end

# Converto il risultato dello scan in indici per la matrice originale (da testare)
function linear_to_cartesian(linear_indices, nrows)
    return [(divrem(idx - 1, nrows) .+ (1, 1))[end:-1:1] for idx in linear_indices]
end

function main()
    print("a")
    vals = Array(Int8[-1, 0, 1, 2, -2, 1, 0, 2])
    has_burning_neibs = KADevice isa CPU ? vals : ROCArray(vals)
    #print(typeof(has_burning_neibs))
    print("b")
    result = gpu_findall_positive!(has_burning_neibs)
    KernelAbstractions.synchronize(KADevice)
    print("c")
    print(Array(result))
end

#TODO: Passare Groupsize da parametro dove serve e con log2 in scan_local_kernel! (problemi con le tuple e isless)

main()