#ENV["JULIA_DEBUG"] = "AMDGPU"
using KernelAbstractions
using AMDGPU
using Cthulhu


if true
    const KADevice = ROCBackend()
else
    const KADevice = CPU()
end

const GROUP_SIZE = 16 #lws


@kernel function mark_positive_kernel!(input, flags, N)
    i = @index(Global)
    if i <= N
        flags[i] = input[i] > 0 ? 1 : 0
    end
end



@kernel function gpu_scan_hybrid!(flags, scan_out, N)
    i = @index(Global)
    lidx = @index(Local)
    
    # per piccoli gruppi
    if GROUP_SIZE <= 32
        val = (i <= N) ? Int32(flags[i]) : Int32(0)
        
        sum = Int32(0)
        for j in Int32(1):(lidx-Int32(1))
            global_j = i - lidx + j
            if global_j <= N && global_j >= Int32(1)
                sum += Int32(flags[global_j])
            end
        end
        sum += val
        
        if i <= N
            scan_out[i] = sum
        end
    else
        #Fallback per gruppi grandi
        temp = @localmem(Int32, GROUP_SIZE)
        
        if i <= N
            temp[lidx] = Int32(flags[i])
        else
            temp[lidx] = Int32(0)
        end
        
        @synchronize()
        
        temp[lidx] += (lidx > Int32(1)) ? temp[lidx - Int32(1)] : Int32(0)
        temp[lidx] += (lidx > Int32(2)) ? temp[lidx - Int32(2)] : Int32(0)
        temp[lidx] += (lidx > Int32(4)) ? temp[lidx - Int32(4)] : Int32(0)
        temp[lidx] += (lidx > Int32(8)) ? temp[lidx - Int32(8)] : Int32(0)
        
        @synchronize()
        
        if i <= N
            scan_out[i] = temp[lidx]
        end
    end
end

@kernel function extract_positions!(flags, scan_out, output_indices, N)
    i = @index(Global)
    
    if i <= N && flags[i] == Int8(1)
        output_pos = scan_out[i]
        if output_pos > 0
            output_indices[output_pos] = Int32(i)
        end
    end
end

# Funzione wrapper
function gpu_findall_positive!(input)
    N = length(input)
    
    flags = KernelAbstractions.zeros(KADevice, Int8, N)
    scan_out = KernelAbstractions.zeros(KADevice, Int32, N)

    #Costruisce flags
    global_size = cld(N, GROUP_SIZE) * GROUP_SIZE
    mark_positive_kernel!(KADevice, global_size, GROUP_SIZE)(input, flags, N)
    KernelAbstractions.synchronize(KADevice)

    #Prefix scan 
    gpu_scan_warp!(KADevice, global_size, GROUP_SIZE)(flags, scan_out, N)
    KernelAbstractions.synchronize(KADevice)
    
    total_count = Array(scan_out[N:N])[1]  # Ultimo elemento del scan
    println("Trovati $total_count elementi positivi")
    
    if total_count == 0
        return KernelAbstractions.zeros(KADevice, Int32, 0)
    end
    
    output_indices = KernelAbstractions.zeros(KADevice, Int32, total_count)
    
    #Estrae le posizioni
    extract_positions!(KADevice, global_size, GROUP_SIZE)(flags, scan_out, output_indices, N)
    KernelAbstractions.synchronize(KADevice)
    
    return output_indices
end

function gpu_findall_positive_debug!(input)
    N = length(input)
    
    flags = KernelAbstractions.zeros(KADevice, Int8, N)
    scan_out = KernelAbstractions.zeros(KADevice, Int32, N)

    global_size = cld(N, GROUP_SIZE) * GROUP_SIZE
    mark_positive_kernel!(KADevice, global_size, GROUP_SIZE)(input, flags, N)
    KernelAbstractions.synchronize(KADevice)
    
    println("Input: ", Array(input))
    println("Flags: ", Array(flags))

    gpu_scan_hybrid!(KADevice, global_size, GROUP_SIZE)(flags, scan_out, N)
    KernelAbstractions.synchronize(KADevice)
    
    println("Scan:  ", Array(scan_out))
    
    total_count = Array(scan_out[N:N])[1]
    println("Count: ", total_count)
    
    if total_count > 0
        output_indices = KernelAbstractions.zeros(KADevice, Int32, total_count)
        extract_positions!(KADevice, global_size, GROUP_SIZE)(flags, scan_out, output_indices, N)
        KernelAbstractions.synchronize(KADevice)
        
        println("Positions: ", Array(output_indices))
        return output_indices
    end
    
    return KernelAbstractions.zeros(KADevice, Int32, 0)
end

function main()
    print("a")
    vals = Array(Int8[-1, 0, 1, 2, -2, 1, 0, 2])
    has_burning_neibs = KADevice isa CPU ? vals : ROCArray(vals)
    print("b")
    result = gpu_findall_positive_debug!(has_burning_neibs)
    KernelAbstractions.synchronize(KADevice)
    print("c")
    print(Array(result))
end

main()