from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

print('mpi_rank = {}, mpi_size = {}'.format(mpi_rank, mpi_size))

mpi_comm.Barrier()
MPI.Finalize()
