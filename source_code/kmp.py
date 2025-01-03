import sys, time
import numpy as np
import multiprocessing as mp
from kguesser import KG_test

def multi_f(bs, be, dat, bk):
    """Function for each process."""
    d, k = KG_test().sim(be-bs, "", "", False)
    print(f"[{bs}-{be}] writing...                              ", end="\r")
    for i, row in enumerate(d):
        dat[bs+i], bk[bs+i] = row, k[i]

def _get_batch(n):
    """Returns the number of loops and batch size based on 'n'."""
    nc = int(mp.cpu_count()/2) if mp.cpu_count() >= 2 else 1
    b = int(n/nc); l = int(b/100) if b > 100 else 1
    b = int(b*2/l) if l > 1 else b; l = int(n/(b*nc))+1
    return b, b*nc, l, nc
def _loop(df, kf, kcol, b, bl, n_cores, ln):
    """Handles the loop in 'multi_sim'."""
    l_proc = []
    with mp.Manager() as manager:
        dat = manager.list(range(ln))
        bk = manager.list(range(ln))
        for c in range(n_cores):
            bs, be = b*c, b*(c+1) if b*(c+1) <= ln else ln
            if bs >= be:
                break
            l_proc.append(mp.Process(target=multi_f, args=(bs, be, dat, bk)))
            l_proc[-1].start()
        for p in l_proc:
            p.join()
        print(f"Saving {ln} datapoints...                         ", end="\r")
        KG_test().save(df, np.array(dat), columns=KG_test().head)
        KG_test().save(kf, np.array(bk), columns=kcol)
def multi_sim(n=10000, df="data.xlsx", kf="k.xlsx"):
    """Call multiple processes with shared lists 'dat' and 'bk'.
    /!\\ Files only written at the end."""
    kcol = ['best_k', 'best_acc']+[str(i) for i in range(1, 101)]
    b, bl, loops, n_cores = _get_batch(n); s = time.time()
    print(f"Starting {loops} loop(s) of {n_cores} batches of {b} rows...")
    for i in range(loops):
        gbs, gbe = bl*i, bl*(i+1) if bl*(i+1) <= n else n
        if gbs >= gbe:
            break
        print(f"Loop {i+1}: [{gbs}-{gbe}]...                      ")
        _loop(df, kf, kcol, b, bl, n_cores, gbe-gbs)
    print(f"Processed {n} rows in {time.time()-s:.02f}s "+
          f"on {mp.cpu_count()} cores.")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    multi_sim(n)
    