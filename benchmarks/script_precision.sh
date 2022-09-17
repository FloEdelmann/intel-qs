#!/bin/bash
#clear

echo -e "\n###################################################################"
echo -e   "##### Precision difference of Intel Quantum Simulator #############"
echo -e   "###################################################################"

##########################################################################

# The script below is run for a system that uses slurm.
# Options are: "pcl-clx", "single-node".
flag_for_slurm="single-node"

# declare -a set_rank_values=(1 2 4 8 16)
declare -a set_rank_values=(1 2 4)

##########################################################################

echo -e "\n -- Setting the parameters that stay unchanged -- \n"

num_threads_per_rank=2

date=$(date +%Y-%m-%d_%H-%M-%S)

out_filename_root="basic_strong_scaling"
out_directory="output/"$date"_"$out_filename_root"/"
num_qubits=10
num_gates=1

##########################################################################

# If the script is launched in a slurm machine.
# Here we considered 8 nodes with 2 sockets each.
if [ $flag_for_slurm == "pcl-clx" ]; then
	num_ranks_per_node=2
	num_threads_per_rank=28
fi

##########################################################################

exec_args=" -nq "$num_qubits" -ng "$num_gates" -od "$out_directory" -of "$out_filename_root\
" -nt "$num_threads_per_rank

exec_file="./bin/basic_code_for_scaling.exe"

##########################################################################

echo -e "\n -- Create the output folder if not present -- \n"

if [ ! -d $out_directory ]; then
	mkdir $out_directory
else
	# Eliminate the summary files if they exist.
	filename=$out_directory$out_filename_root"_first_q"$num_qubits"_p0.txt"
        echo "% Time cost (in sec) for 1-qubit gate on first qubit vs num_ranks (double precision)" > $filename 
	filename=$out_directory$out_filename_root"_last_q"$num_qubits"_p0.txt"
        echo "% Time cost (in sec) for 1-qubit gate on last qubit vs num_ranks (double precision)" > $filename 
	filename=$out_directory$out_filename_root"_first_q"$num_qubits"_p1.txt"
        echo "% Time cost (in sec) for 1-qubit gate on first qubit vs num_ranks (float precision)" > $filename 
	filename=$out_directory$out_filename_root"_last_q"$num_qubits"_p1.txt"
        echo "% Time cost (in sec) for 1-qubit gate on last qubit vs num_ranks (float precision)" > $filename 
	filename=$out_directory$out_filename_root"_first_q"$num_qubits"_p2.txt"
        echo "% Time cost (in sec) for 1-qubit gate on first qubit vs num_ranks (posit precision)" > $filename 
	filename=$out_directory$out_filename_root"_last_q"$num_qubits"_p2.txt"
        echo "% Time cost (in sec) for 1-qubit gate on last qubit vs num_ranks (posit precision)" > $filename 
fi

##########################################################################

echo -e "\n -- Run the basic executable -- \n"

for num_ranks in "${set_rank_values[@]}"
do
	if [ $flag_for_slurm == "single-node" ]; then
		num_ranks_per_node=$num_ranks
	fi
	mpiexec -n $num_ranks -npernode $num_ranks_per_node -x I_MPI_DEBUG=4 -x OMP_NUM_THREADS=$num_threads_per_rank -x KMP_AFFINITY=granularity=fine $exec_file $exec_args -gp double
	mpiexec -n $num_ranks -npernode $num_ranks_per_node -x I_MPI_DEBUG=4 -x OMP_NUM_THREADS=$num_threads_per_rank -x KMP_AFFINITY=granularity=fine $exec_file $exec_args -gp float
	mpiexec -n $num_ranks -npernode $num_ranks_per_node -x I_MPI_DEBUG=4 -x OMP_NUM_THREADS=$num_threads_per_rank -x KMP_AFFINITY=granularity=fine $exec_file $exec_args -gp posit
done

##########################################################################

# echo -e "\n -- Graph of gate-time per qubit at highest number of ranks -- \n"

# num_ranks=${set_rank_values[-1]}
# data_filename=$out_directory$out_filename_root"_q"$num_qubits"_n"$num_ranks".txt"
# plot_filename=${out_directory}one_qubit_gates.png

# gnuplot_instructions="set xlabel \"qubits\" font \",10\";"\
# "set ylabel \"time (sec)\" font \",10\";"\
# "set style line 1 lc rgb '#8b1a0e' pt 1 ps 1 lt 1 lw 2;"\
# "set style line 2 lc rgb '#5e9c36' pt 6 ps 1 lt 1 lw 2;"\
# "set terminal png size 800,600; set output '$plot_filename';"\
# "plot '$data_filename' using 1:2 title \"$num_ranks ranks\" with linespoints ls 1;"

# gnuplot --persist -e "$gnuplot_instructions"

# echo -e "\n [ plot in file "$plot_filename" ] \n"
# display $plot_filename &

##########################################################################

echo -e "\n -- Graph of gate-time per qubit at highest number of ranks -- \n"

data_filename_first_double=$out_directory$out_filename_root"_first_q"$num_qubits"_p0.txt"
data_filename_last_double=$out_directory$out_filename_root"_last_q"$num_qubits"_p0.txt"
data_filename_first_float=$out_directory$out_filename_root"_first_q"$num_qubits"_p1.txt"
data_filename_last_float=$out_directory$out_filename_root"_last_q"$num_qubits"_p1.txt"
data_filename_first_posit=$out_directory$out_filename_root"_first_q"$num_qubits"_p2.txt"
data_filename_last_posit=$out_directory$out_filename_root"_last_q"$num_qubits"_p2.txt"
plot_filename=${out_directory}precision.png

declare first_rank_x_value=${set_rank_values[0]}-0.5
declare last_rank_x_value=${set_rank_values[-1]}+1

gnuplot_instructions="set xlabel \"num ranks\" font \",10\";"\
"set ylabel \"time (sec)\" font \",10\";"\
"set xtics format \"\" nomirror;"\
"set style line 1 lc rgb '#ff0000' pt 1 ps 1 lt 1 lw 1;"\
"set style line 2 lc rgb '#aa0000' pt 6 ps 1 lt 1 lw 1;"\
"set style line 3 lc rgb '#00ff00' pt 1 ps 1 lt 1 lw 1;"\
"set style line 4 lc rgb '#00aa00' pt 6 ps 1 lt 1 lw 1;"\
"set style line 5 lc rgb '#0000ff' pt 1 ps 1 lt 1 lw 1;"\
"set style line 6 lc rgb '#0000aa' pt 6 ps 1 lt 1 lw 1;"\
"set terminal png size 800,600; set output '$plot_filename';"\
"set xrange [${first_rank_x_value}:${last_rank_x_value}];"\
"xoffset(x)=x+0.5;"\
"plot '$data_filename_first_double' using 1:2:xticlabels(strcol(1).\"\\nd\") title \"first qubit (double)\" with linespoints ls 1,"\
"'$data_filename_last_double' using 1:2 title \"last qubit (double)\" with linespoints ls 2,"\
"'$data_filename_first_float' using (xoffset(\$1)):2:xticlabels(strcol(1).\"\\nf\") title \"first qubit (float)\" with linespoints ls 3,"\
"'$data_filename_last_float' using (xoffset(\$1)):2 title \"last qubit (float)\" with linespoints ls 4,"\
"'$data_filename_first_posit' using (xoffset(\$1)):2:xticlabels(strcol(1).\"\\np\") title \"first qubit (posit)\" with linespoints ls 5,"\
"'$data_filename_last_posit' using (xoffset(\$1)):2 title \"last qubit (posit)\" with linespoints ls 6;"

gnuplot --persist -e "$gnuplot_instructions"

echo -e "\n [ plot in file "$plot_filename" ] \n"
display $plot_filename &

##########################################################################
