from datetime import datetime
import multiprocessing as mp


def run_in_mp_queue(fxn, list_of_args, num_cpus, logger):
    """ Execute fxn on each args tuple in list_of_args over num_cpus processes.
    """

    # Create the process pool and launch processes to deal with it.
    start_dt = datetime.now()
    logger.info(f"Creating MP Queue at {start_dt.strftime('%Y-%m-%d %I:%M')}")

    # Fill the queue with jobs
    task_queue = mp.Queue()
    for argument_tuple in list_of_args:
        task_queue.put(argument_tuple)
    print(f"  queue has {task_queue.qsize()} jobs")
    for _ in range(num_cpus):
        task_queue.put(None)  # to kill each worker when real jobs are complete
    print(f"  queue has {task_queue.qsize()} (jobs + Nones)")

    # Create processes to handle the jobs
    processes = []
    rslt_queue = mp.Queue()
    for pid in range(num_cpus):
        proc = mp.Process(
            target=fxn, args=(task_queue, rslt_queue, pid)
        )
        # proc.daemon = True  # process run in background and clean up its mess
        print(f"  start process {pid}")
        proc.start()
        processes.append(proc)
    # All processes are now running separately.
    # This process will continue without waiting until rslt_queue.get() below.

    print(f"  queue has {rslt_queue.qsize()} results")

    # Results are stuck back onto the queue, so we need to get them.
    # This rslt_queue.get() should wait for a result,
    # pausing this thread until processes finish.
    results = []
    for _ in list_of_args:
        result = rslt_queue.get()
        print(f"  GOT a result")
        results.append(result)

    end_dt = datetime.now()
    logger.info(f"Completed MP Queue at {end_dt.strftime('%Y-%m-%d %I:%M')} "
                f"with {rslt_queue.qsize()} ({len(results)}) results.")

    return results
