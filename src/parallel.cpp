#include "studio.h"
#include <stdlib.h>
#include <mpi.h>

int main (int argc, char *argv[]) {

    int id, nthreads;
    char *cpu_name;
    double time_initial, time_current, time;

    // Launch MPI processes on each node
    MPI_Init(&argc, &argv);
    time_initial = MPI_Wtime();

    // Get the id and number of threads
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &nthreads);

    // on every process allocate space for machine name
    cpu_name = (char *)calloc(80, sizeof(char));
    gethostname(cpu_name, 80);
    time_currrent = MPI_Wtime();
    time = time_current - time_initial;

    printf("hello MPI user: from process = %i ono machine %s, of NCPU=%i process \n", 
    id, cpu_name, nthreads);

    MPI_Finalize();

    return 0;
}