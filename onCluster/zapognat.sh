#!/bin/bash

i=0
while read p
do
	sbatch skripta.sh $p
done < seedi.txt
