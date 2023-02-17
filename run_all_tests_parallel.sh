#! /bin/bash

N=3  # Amount of jobs to run in parallel
T=0  # Counter for amount of jobs
Q=() # Job queue
FILE='jobs.txt'

# Clean Q array
function _clean {
        for ((i=0; i < ${N}; ++i)); do
                tst=/proc/${Q[$i]}
                if [ ! -d $tst ]; then
                        echo 'Completed: $tst'
                        Q[$i]=0
                fi
        done
}

# Setup the Q
for ((i=0; i < $N; i++)); do
        Q[$i]=0
done

while read -r line; do
        echo $line
        $line &

        # Try to find an open sport (Q[i]=0)
        while true; do
                for ((i=0; i < ${N}; ++i)); do
                        if [ ${Q[$i]} -eq 0 ]; then
                                Q[$i]=$!
                                break 2
                        fi
                done
                # Clean the Q array if no free entry is found
                _clean
        done
        ((T++))

done < ${FILE}
wait
echo "Processed ($T/$(wc -l < jobs.txt)) jobs"
exit 0