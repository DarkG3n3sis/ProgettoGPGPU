using AMDGPU, ArgParse


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "-d"
            help = "a positional argument"
            required = true
    end

    return parse_args(s)
end


function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (d,val) in parsed_args
        println("  $arg  =>  $val")
    end


    #Variables definition
    ncols::Int = size(d, 1)
    nrows::Int = size(d, 2)

    println("cols:"$ncols, "raws:" $nrows)
end

main()