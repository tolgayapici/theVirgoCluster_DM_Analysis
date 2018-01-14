#!/bin/env bash

masses=( 1000.0         1668.1005372   2782.55940221
	 4641.58883361  7742.63682681  12915.49665015
	 21544.34690032 35938.13663805 59948.42503189
	 100000. )
#channels=( 1 2 3 4 5 )
channels=( 4 )

cd ../py

for channel in ${channels[@]}; do
    for mass in ${masses[@]}; do
	echo "running for mass ${mass} GeV and channel ${channel}"
	python VirgoCluster_linked.py ${mass} ${channel} 1 GAO 1 VERITAS &> ../results/decay_mass${mass}GeV_channel${channel}_GAO.log
	python VirgoCluster_linked.py ${mass} ${channel} 1 B01 1 VERITAS &> ../results/decay_mass${mass}GeV_channel${channel}_B01.log
    done
done
