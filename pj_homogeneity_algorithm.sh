#!/bin/sh
homogeneity_algorithm()
{
	for i in {1..100}
	do
		python3 /geir_data/scr/ccheng/AST425/Personal/pj_run_code.py --cluster=$1 --num_sigma=$2 --red_clump=$3 --run_number=${i} --location=$4 --elem=$5 &
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

mkdir run_files_${a}_${e}
cd run_files_${a}_${e}
cp /geir_data/scr/ccheng/AST425/Personal/pj_clusters_input.py /geir_data/scr/ccheng/AST425/Personal/run_files_${a}_${e}/pj_clusters_input.py
cp /geir_data/scr/ccheng/AST425/Personal/occam_clusters_post_process.py /geir_data/scr/ccheng/AST425/Personal/run_files_${a}_${e}/occam_clusters_post_process.py
cp /geir_data/scr/ccheng/AST425/Personal/pj_ABC.py /geir_data/scr/ccheng/AST425/Personal/run_files_${a}_${e}/pj_ABC.py
cp /geir_data/scr/ccheng/AST425/Personal/pj_run_code.py /geir_data/scr/ccheng/AST425/Personal/run_files_${a}_${e}/pj_run_code.py
mkdir /geir_data/scr/ccheng/AST425/Personal/run_files_${a}_${e}/${a}

homogeneity_algorithm $a $b $c $d $e