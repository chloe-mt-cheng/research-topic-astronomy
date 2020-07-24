#!/bin/sh
homogeneity_algorithm()
{
	for i in 1 2 3 4 5 6 7 8 9 10
	do
		python3 /geir_data/scr/ccheng/AST425/Personal/run_code.py --cluster=$1 --num_sigma=$2 --red_clump=$3 --run_number=${i} --location=$4 --elem=$5 &
	done
}

echo "Enter a cluster:"
read a
echo "Enter the number of simulations:"
read b
echo "Enter True if you want to remove red clump stars, False if not:"
read c
echo "Enter the run location:"
read d
echo "Enter an element:"
read e

mkdir run_files
cd run_files 
cp /geir_data/scr/ccheng/AST425/Personal/occam_clusters_input.py /geir_data/scr/ccheng/AST425/Personal/run_files/occam_clusters_input.py
cp /geir_data/scr/ccheng/AST425/Personal/occam_clusters_post_process.py /geir_data/scr/ccheng/AST425/Personal/run_files/occam_clusters_post_process.py
cp /geir_data/scr/ccheng/AST425/Personal/ABC.py /geir_data/scr/ccheng/AST425/Personal/run_files/ABC.py
cp /geir_data/scr/ccheng/AST425/Personal/run_code.py /geir_data/scr/ccheng/AST425/Personal/run_files/run_code.py
mkdir /geir_data/scr/ccheng/AST425/Personal/run_files/${a}

homogeneity_algorithm $a $b $c $d $e
