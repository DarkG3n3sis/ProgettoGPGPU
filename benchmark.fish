#!/usr/bin/env fish

set binary $argv[1]
if test -z "$binary"
    set binary "findall_lmem_v2"
end

echo "=== Testing binary: $binary ==="

echo "=== CPU: Scalabilità N (lws=256) ==="
for e in 10 14 16 18 20 22 24
    set n (math 2^$e)
    echo -n "N=2^$e ($n): "
    OCL_PLATFORM=1 "./"$binary 256 $n 2>/dev/null | grep -i "total:"; or echo "(N too large for this binary)"
end

echo ""
echo "=== GPU: Scalabilità N (lws=256) ==="
for e in 10 14 16 18 20 22 24
    set n (math 2^$e)
    echo -n "N=2^$e ($n): "
    OCL_PLATFORM=0 "./"$binary 256 $n 2>/dev/null | grep -i "total:"; or echo "(N too large for this binary)"
end

echo ""
echo "=== GPU vs CPU (N=2^24, vari lws) ==="
for lws in 256 512 1024
    echo -n "GPU lws=$lws: "
    OCL_PLATFORM=0 "./"$binary $lws (math 2^24) 2>/dev/null | grep -i "total:"; or echo "(N too large for this binary)"
    echo -n "CPU lws=$lws: "
    OCL_PLATFORM=1 "./"$binary $lws (math 2^24) 2>/dev/null | grep -i "total:"; or echo "(N too large for this binary)"
end
