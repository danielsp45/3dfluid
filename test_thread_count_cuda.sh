# Define the block dimension combinations to test (x, y, z)
BLOCK_DIMS=(
    "8 4 1 32"     # 32 threads
    "16 2 1 32"    # 32 threads
    "32 1 1 32"    # 32 threads

    "8 8 1 64"     # 64 threads
    "16 4 1 64"    # 64 threads
    "32 2 1 64"    # 64 threads
    "64 1 1 64"    # 64 threads

    "16 8 1 128"    # 128 threads
    "32 4 1 128"    # 128 threads
    "64 2 1 128"    # 128 threads
    "128 1 1 128"   # 128 threads

    "16 16 1 256"   # 256 threads
    "32 8 1 256"    # 256 threads
    "64 4 1 256"    # 256 threads
    "128 2 1 256"   # 256 threads
    "256 1 1 256"   # 256 threads

    "32 16 1 512"   # 512 threads
    "64 8 1 512"    # 512 threads
    "128 4 1 512"   # 512 threads
    "256 2 1 512"   # 512 threads
    "512 1 1 512"   # 512 threads

    "32 32 1 1024"   # 1024 threads
    "64 16 1 1024"   # 1024 threads
    "128 8 1 1024"   # 1024 threads
    "256 4 1 1024"   # 1024 threads
    "512 2 1 1024"   # 1024 threads
    "1024 1 1 1024"  # 1024 threads
)

# Iterate over each block dimension combination
for index in "${!BLOCK_DIMS[@]}"; do
    DIM="${BLOCK_DIMS[$index]}"

    # Read block dimensions
    read BLOCK_X BLOCK_Y BLOCK_Z THREADS_PER_BLOCK <<< "$DIM"

    echo "----------------------------------------"
    echo "Testing block dimensions: ($BLOCK_X, $BLOCK_Y, $BLOCK_Z) $THREADS_PER_BLOCK"

    # Clean previous builds
    make clean

    # Compile the CUDA code with current block dimensions and executable name
    make LIN_SOLVE_BLOCK_X=$BLOCK_X LIN_SOLVE_BLOCK_Y=$BLOCK_Y LIN_SOLVE_BLOCK_Z=$BLOCK_Z THREADS_PER_BLOCK=$THREADS_PER_BLOCK

    make rrun
    make rrun
    make rrun
done

echo "----------------------------------------"
echo "All tests completed."