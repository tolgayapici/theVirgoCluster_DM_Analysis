#!/bin/env bash                        

masses=(  1000.             1274.2749857     1623.77673919                    
          2069.13808111     2636.65089873    3359.81828628                    
          4281.33239872     5455.59478117    6951.92796178                    
          8858.6679041      11288.37891685   14384.49888288                   
          18329.80710832    23357.2146909    29763.51441631                   
          37926.90190732    48329.30238572   61584.8211066                    
          78475.99703515    100000. )  

channels=( 1 2 3 4 5 )                 

for channel in ${channels[@]}; do
    channel_jobs="jobs/channel-${channel}.sh"
    for mass in ${masses[@]}; do       
        sed "s/%%%MASS%%%/${mass}/g" template_job.sh | sed "s/%%%CHANNEL%%%/${channel}/g" > jobs/mass-${mass}-channel-${channel}.sh
	echo "sbatch jobs/mass-${mass}-channel-${channel}.sh" >> ${channel_jobs}
    done
done
