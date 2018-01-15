#!/bin/env bash

masses=( 1000.00  1668.10  2782.56
	 4641.59  7742.64  12915.50
	 21544.35 35938.14 59948.43
	 100000.00 )

#channels=( 1 2 3 4 5 )
channels=( 4 )

cd ../py

for channel in ${channels[@]}; do
    for mass in ${masses[@]}; do
	echo "running for mass ${mass} GeV and channel ${channel}"
	time python VirgoCluster_annihilation_extended.py ${mass} ${channel} GAO 1 VERITAS&>../results/annihilation_mass${mass}GeV_channel${channel}_Sanchez.log
	#python VirgoCluster_annihilation_extended.py ${mass} ${channel} B01 1 VERITAS&>../results/annihilation_mass${mass}GeV_channel${channel}_B01.log
    done
done
