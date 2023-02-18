from func import f
import multiprocessing as mp

if __name__ == '__main__':
    lst = [1,2,3]
    pool = mp.Pool(24)
    print(pool.map(f, lst))
