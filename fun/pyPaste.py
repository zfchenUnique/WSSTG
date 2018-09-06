def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


from tqdm import tqdm

################################################################################
def apply_function(func_to_apply, queue_in, queue_out):
    while not queue_in.empty():
        num, obj = queue_in.get()
        queue_out.put((num, func_to_apply(obj)))

################################################################################
def prll_map(func_to_apply, items, cpus=None, verbose=False):
    # Number of processes to use #
    if cpus is None: cpus = min(multiprocessing.cpu_count(), 32)
    # Create queues #
    q_in  = multiprocessing.Queue()
    q_out = multiprocessing.Queue()
    # Process list #
    new_proc  = lambda t,a: multiprocessing.Process(target=t, args=a)
    processes = [new_proc(apply_function, (func_to_apply, q_in, q_out)) for x in range(cpus)]
    # Put all the items (objects) in the queue #
    sent = [q_in.put((i, x)) for i, x in enumerate(items)]
    # Start them all #
    for proc in processes:
        proc.daemon = True
        proc.start()
    # Display progress bar or not #
    if verbose:
        results = [q_out.get() for x in tqdm(range(len(sent)))]
    else:
        results = [q_out.get() for x in range(len(sent))]
    # Wait for them to finish #
    for proc in processes: proc.join()
    # Return results #
    return [x for i, x in sorted(results)]

def apply_func(f, q_in, q_out):
    while not q_in.empty():
        i, x = q_in.get()
        q_out.put((i, f(x)))

# map a function using a pool of processes
def parmap(f, X, nprocs = cpu_count()):
    q_in, q_out   = Queue(), Queue()
    proc = [Process(target=apply_func, args=(f, q_in, q_out)) for _ in range(nprocs)]
    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [p.start() for p in proc]
    res = [q_out.get() for _ in sent]
    [p.join() for p in proc]

    return [x for i,x in sorted(res)]



