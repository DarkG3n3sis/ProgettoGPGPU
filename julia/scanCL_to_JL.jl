using KernelAbstractions
using AMDGPU

if true
    const KADevice = ROCBackend()
else
    const KADevice = CPU()
end


const GROUP_SIZE = 512


@kernel function scan_kernel!(out, tails, input, nels)
    i = @index(Global)
    li = @index(Local)
    group_sz = @groupsize()
    group_sz = group_sz[1]

    temp = @localmem(Int32, GROUP_SIZE)

    val = ntuple(j -> Int32(0), 4)
    if i < nels
        val = input[i]
        # Simula il comportamento:
        # val.s1 += val.s0 ; val.s3 += val.s2
        val = (val[1],
               val[2] + val[1],
               val[3],
               val[4] + val[3])
        # val.s3 += val.s2 ; quindi s3 = val[4] + val[3]
        val = (val[1], val[2], val[3], val[4] + val[3])
    end

    temp[li+1] = val[4]  # offset +1 per Julia 1-based indexing

    @synchronize()

    active_mask = 1
    while active_mask < group_sz
        pull_mask = li & (~(active_mask - 1))
        pull_mask -= 1
        @synchronize()
        if (li & active_mask) != 0 && pull_mask >= 0
            temp[li+1] += temp[pull_mask+1]
        end
        active_mask *= 2
    end

    @synchronize()

    if li > 0
        prefix = temp[li]
        val = (val[1] + prefix,
               val[2] + prefix,
               val[3] + prefix,
               val[4] + prefix)
    end

    if i < nels
        out[i] = val
    end

    if @index(Local) == 0 && @ndrange()[1] > 1
        tails[@ndrange()[2]] = temp[group_sz]
    end
end


@kernel function scan_fixup!(out, tails, nels)
    i = @index(Global)
    group = @ndrange()[2]

    if group > 0 && i < nels
        fixup = tails[group - 1]
        val = out[i]
        out[i] = (val[1] + fixup,
                  val[2] + fixup,
                  val[3] + fixup,
                  val[4] + fixup)
    end
end


function main()
    # Parametri iniziali
    nels = 512
    n_groups = cld(nels, GROUP_SIZE)

    # Input array: ogni elemento è un NTuple{4, Int32}
    input_host = [(Int32(i), Int32(1), Int32(0), Int32(2)) for i in 0:nels-1]
    input = ROCArray(input_host)

    # Output: stessa dimensione
    output = ROCArray{NTuple{4, Int32}}(undef, nels)

    # Array per i "tails" dei gruppi
    tails = ROCArray{Int32}(undef, n_groups)

    # Kernel scan parziale
    event1 = scan_kernel!(KADevice, nels, GROUP_SIZE)(
        output, tails, input, nels; ndrange=nels
    )
    KernelAbstractions.synchronize(KADevice)

    # Fixup: correzione con somma delle tails
    event2 = scan_fixup!(KADevice, nels, GROUP_SIZE)(
        output, tails, nels; ndrange=nels
    )
    KernelAbstractions.synchronize(KADevice)

    # Portiamo su CPU per ispezione
    result = collect(output)
    for i in 1:min(nels, 20)
        println("[$i] => ", result[i])
    end
end

main()