import datetime
import pickle

print("Current system time:", datetime.datetime.now())

ncores = 2
print(ncores)
RUN_LEVEL = 1
match RUN_LEVEL:
    case 1:
        NP_FITR = 2
        NFITR = 2
        NREPS_FITR = ncores
        NP_EVAL = 2
        NREPS_EVAL = ncores
        NREPS_EVAL2 = ncores
        print("Running at level 1")
    case 2:
        NP_FITR = 5000
        NFITR = 100
        NREPS_FITR = ncores
        NP_EVAL = 10000
        NREPS_EVAL = ncores
        NREPS_EVAL2 = ncores*8
        print("Running at level 2")


DATA = pickle.load(open('../input/ur_measles.pkl', 'rb'))
MODEL = None

