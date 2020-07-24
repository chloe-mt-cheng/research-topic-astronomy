#!/bin/sh
homogeneity_algorithm() ###Function to run the algorithm several times
{
	for i in 1 2 3 4 5 6 7 8 9 10 ###Loop through and run 10 times
	do ###Run run_code.py 10 times 
		python3 /geir_data/scr/ccheng/AST425/Personal/run_code.py --cluster=$1 --num_sigma=$2 --red_clump=$3 --run_number=${i} --location=$4 --elem=$5 &
	done
}

echo "Enter a cluster:" ###Input the name of the cluster (no quotes, no spaces)
read a
echo "Enter the number of simulations:" ###Enter the number of simulations you want to run
read b
echo "Enter True if you want to remove red clump stars, False if not:" ###Enter whether red clumps are to be removed or not
read c
echo "Enter the run location:" ###Enter whether running on personal or server
read d
echo "Enter an element:" ###Enter the name of an element
read e

mkdir run_files ###Make a directory called run_files
cd run_files ###cd into it and move all scripts into it to run just in that directory
cp /geir_data/scr/ccheng/AST425/Personal/occam_clusters_input.py /geir_data/scr/ccheng/AST425/Personal/run_files/occam_clusters_input.py
cp /geir_data/scr/ccheng/AST425/Personal/occam_clusters_post_process.py /geir_data/scr/ccheng/AST425/Personal/run_files/occam_clusters_post_process.py
cp /geir_data/scr/ccheng/AST425/Personal/ABC.py /geir_data/scr/ccheng/AST425/Personal/run_files/ABC.py
cp /geir_data/scr/ccheng/AST425/Personal/run_code.py /geir_data/scr/ccheng/AST425/Personal/run_files/run_code.py
mkdir /geir_data/scr/ccheng/AST425/Personal/run_files/${a} ###Make another subdirectory named after the cluster

homogeneity_algorithm $a $b $c $d $e ###Run the function 
