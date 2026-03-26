#!/usr/bin/env fish

echo "=== CPU: Scalabilità N (lws=256) ==="
for e in 10 14 16 18 20 22 24 26 28
    set n (math 2^$e)
    echo -n "N=2^$e ($n): "
    OCL_PLATFORM=1 ./findall 256 $n 2>/dev/null | grep "total:"
end

echo ""
echo "=== GPU: Scalabilità N (lws=256) ==="
for e in 10 14 16 18 20 22 24 26 27
    set n (math 2^$e)
    echo -n "N=2^$e ($n): "
    OCL_PLATFORM=0 ./findall 256 $n 2>/dev/null | grep "total:"
end

echo ""
echo "=== GPU vs CPU (N=2^24, vari lws) ==="
for lws in 32 64 128 256 512 1024
    echo -n "GPU lws=$lws: "
    OCL_PLATFORM=0 ./findall $lws (math 2^24) 2>/dev/null | grep "total:"
    echo -n "CPU lws=$lws: "
    OCL_PLATFORM=1 ./findall $lws (math 2^24) 2>/dev/null | grep "total:"
end