
for n in 100 200 500 1000
do
    for rho in 0.0 0.2
    do
        for p in 20 50 100
        do
        	echo "#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-4:00			#job run 12 hour
#SBATCH -p short			#submit to 'short' queue
#SBATCH --mem=12000  		# use 4 GB memory
#SBATCH -o wd45_%j.out
#SBATCH -e wd45_%j.err
#SBATCH --mail-type=END      #Type of email notification
#SBATCH --mail-user=wdeng@hsph.harvard.edu
module load gcc/6.2.0 python/3.6.0
python temp0.py $n $rho $p" > job_${n}_${rho}_${p}.sh
			
			sbatch job_${n}_${rho}_${p}.sh
            #rm job_${n}_${rho}_${p}.sh
        
        done
    done
done
