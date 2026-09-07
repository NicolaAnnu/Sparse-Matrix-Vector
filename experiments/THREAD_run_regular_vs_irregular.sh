#!/bin/bash
set -euo pipefail
source ./common_config.sh

N=$DEFAULT_N
NZ=$DEFAULT_NZ
SEED=$DEFAULT_SEED

MODES=("regular" "irregular")
THREAD_COUNTS=(16 32)
BLOCK_SIZES=(1024 2048 4096)

BASELINE_FILE="$RESULTS_DIR/THREAD_baseline_speedup.csv"
LEGACY_OUT="$RESULTS_DIR/THREAD_regular_vs_irregular.csv"
SWEEP_OUT="$RESULTS_DIR/THREAD_regular_vs_irregular_sweep.csv"

# Preserve any already available sequential baselines before overwriting outputs.
declare -A SEQ_BASELINES

for mode in "${MODES[@]}"; do
    seq_med=""

    # First try the general baseline file (normally contains the irregular case).
    if [[ -f "$BASELINE_FILE" ]]; then
        seq_med=$(awk -F, \
            -v n="$N" \
            -v nz="$NZ" \
            -v mode="$mode" \
            -v seed="$SEED" \
            'NR > 1 && $1 == n && $2 == nz && $3 == mode && $4 == seed {
                print $7
                exit
            }' "$BASELINE_FILE")
    fi

    # Then try a previously generated regular-vs-irregular result.
    if [[ -z "$seq_med" && -f "$LEGACY_OUT" ]]; then
        seq_med=$(awk -F, \
            -v n="$N" \
            -v nz="$NZ" \
            -v mode="$mode" \
            -v seed="$SEED" \
            'NR > 1 && $1 == n && $2 == nz && $3 == mode && $4 == seed {
                print $7
                exit
            }' "$LEGACY_OUT")
    fi

    # If no baseline exists, calculate it once for this mode.
    if [[ -z "$seq_med" ]]; then
        echo "Calculating sequential baseline for mode=$mode"
        seq_times=()

        for r in $(seq 1 "$REPEATS"); do
            echo "Sequential $mode run $r/$REPEATS"

            output=$(run_seq \
                -n "$N" \
                -nz "$NZ" \
                -m "$mode" \
                -s "$SEED")

            echo "$output"
            t=$(echo "$output" | extract_time)

            if [[ -z "$t" ]]; then
                echo "Error: could not extract sequential execution time for mode=$mode" >&2
                exit 1
            fi

            seq_times+=("$t")
        done

        seq_med=$(calculate_median "${seq_times[@]}")
    else
        echo "Sequential baseline recovered for mode=$mode: $seq_med s"
    fi

    SEQ_BASELINES[$mode]="$seq_med"
done

# Keep the original file/schema used by zplot_results.py.
# It contains only block_size=1024, exactly as before.
echo "n,nz,mode,seed,threads,block_size,seq_time_med,thread_time_med,speedup,efficiency_percent" > "$LEGACY_OUT"

# Complete sweep requested for the regular-vs-irregular comparison.
echo "n,nz,mode,seed,threads,block_size,seq_time_med,thread_time_med,speedup,efficiency_percent" > "$SWEEP_OUT"

for mode in "${MODES[@]}"; do
    seq_med="${SEQ_BASELINES[$mode]}"

    for threads in "${THREAD_COUNTS[@]}"; do
        for block_size in "${BLOCK_SIZES[@]}"; do
            echo
            echo ">> C++ Threads: mode=$mode threads=$threads block=$block_size (repeats=$REPEATS)"

            times=()

            for r in $(seq 1 "$REPEATS"); do
                echo "Run $r/$REPEATS"

                output=$(run_cpp_threads "$threads" \
                    -n "$N" \
                    -nz "$NZ" \
                    -m "$mode" \
                    -s "$SEED" \
                    -b "$block_size")

                echo "$output"
                t=$(echo "$output" | extract_time)

                if [[ -z "$t" ]]; then
                    echo "Error: could not extract C++ Threads execution time" >&2
                    exit 1
                fi

                times+=("$t")
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

            row="$N,$NZ,$mode,$SEED,$threads,$block_size,$seq_med,$thread_med,$speedup,$efficiency"
            echo "$row" >> "$SWEEP_OUT"

            # Preserve the legacy result expected by zplot_results.py.
            if [[ "$block_size" -eq 1024 ]]; then
                echo "$row" >> "$LEGACY_OUT"
            fi

            echo "  -> Median Total Time: ${thread_med}s"
        done
    done
done

echo
echo "Legacy results (block_size=1024): $LEGACY_OUT"
echo "Complete sweep: $SWEEP_OUT"
