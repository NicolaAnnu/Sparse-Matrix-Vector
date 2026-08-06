#!/bin/bash
set -euo pipefail
source ./common_config.sh

N=$DEFAULT_N
NZ=$DEFAULT_NZ
MODE=$DEFAULT_MODE
SEED=$DEFAULT_SEED
BLOCK_SIZE=$DEFAULT_BLOCK_SIZE
THREAD_COUNTS=(1 2 4 8 16 32)

BASELINE_FILE="$RESULTS_DIR/baseline_speedup.csv"
OUT="$RESULTS_DIR/thread_scaling.csv"

# Controlla che la baseline sia già stata calcolata
if [[ ! -f "$BASELINE_FILE" ]]; then
    echo "Errore: manca $BASELINE_FILE"
    echo "Esegui prima ./run_baseline_speedup.sh"
    exit 1
fi

# Recupera il tempo sequenziale con gli stessi N, NZ, mode e seed
seq_med=$(awk -F, \
    -v n="$N" \
    -v nz="$NZ" \
    -v mode="$MODE" \
    -v seed="$SEED" \
    'NR > 1 && $1 == n && $2 == nz && $3 == mode && $4 == seed {
        print $7
        exit
    }' "$BASELINE_FILE")

if [[ -z "$seq_med" ]]; then
    echo "Errore: nel CSV della baseline non esiste una configurazione compatibile."
    exit 1
fi

echo "Sequential median recuperata: $seq_med s"

echo "n,nz,mode,seed,block_size,threads,seq_time_med,thread_time_med,speedup,efficiency_percent" > "$OUT"

for threads in "${THREAD_COUNTS[@]}"; do
    echo "Testing $threads threads"

    times=()

    for r in $(seq 1 "$REPEATS"); do
        echo "Run $r/$REPEATS"

        output=$(run_cpp_threads "$threads" \
            -n "$N" \
            -nz "$NZ" \
            -m "$MODE" \
            -s "$SEED" \
            -b "$BLOCK_SIZE")

        echo "$output"
        times+=("$(echo "$output" | extract_time)")
    done

    thread_med=$(calculate_median "${times[@]}")

    speedup=$(awk \
        -v s="$seq_med" \
        -v p="$thread_med" \
        'BEGIN {printf "%.6f", s/p}')

    efficiency=$(awk \
        -v s="$speedup" \
        -v t="$threads" \
        'BEGIN {printf "%.4f", 100*s/t}')

    echo "$N,$NZ,$MODE,$SEED,$BLOCK_SIZE,$threads,$seq_med,$thread_med,$speedup,$efficiency" >> "$OUT"
done

echo "Results: $OUT"