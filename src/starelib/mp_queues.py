from datetime import datetime
import multiprocessing as mp


def queue_consumer(task_q, rslt_q, pid, fxn, logger):
    logger.debug(f"  process {pid} is alive and checking the queue.")
    while True:
        # Get the next set of parameters to minimize
        msg = task_q.get()
        if msg is None:
            break
        else:
            # Save results to the result queue,
            logger.debug(f"  process {pid} sending worker tuple "
                         f"for region {msg[0]}...")
            rslt_q.put(fxn(msg))

    logger.debug(f"  process {pid} found an empty queue and is exiting.")


def run_in_mp_queue(fxn, list_of_args, num_cpus, logger):
    """ Execute fxn on each args tuple in list_of_args over num_cpus processes.
    """

    # Create the process pool and launch processes to deal with it.
    start_dt = datetime.now()
    logger.info(f"Creating MP Queue with {num_cpus} processors "
                f"at {start_dt.strftime('%Y-%m-%d %I:%M')}")
    for handler in logger.handlers:
        handler.flush()

    # Fill the queue with jobs
    task_queue = mp.Queue()
    for argument_tuple in list_of_args:
        task_queue.put(argument_tuple)
    logger.debug(f"  queue has {task_queue.qsize()} jobs")
    for _ in range(num_cpus):
        task_queue.put(None)  # to kill each worker when real jobs are complete
    logger.debug(f"  queue has {task_queue.qsize()} (jobs + Nones)")

    # Create processes to handle the jobs
    processes = []
    rslt_queue = mp.Queue()
    for proc_id in range(num_cpus):
        proc = mp.Process(
            # name="some unique name not 'Process-1'",
            target=queue_consumer,
            args=(task_queue, rslt_queue, proc_id, fxn, logger)
        )
        # proc.daemon = True  # process run in background and clean up its mess
        logger.debug(f"  start process {proc_id}")
        proc.start()
        processes.append(proc)
    # All processes are now running separately.
    # This process will continue without waiting until rslt_queue.get() below.

    logger.debug(f"  queue has {rslt_queue.qsize()} results")

    # Results are stuck back onto the queue, so we need to get them.
    # This rslt_queue.get() should wait for a result,
    # pausing this thread until processes finish.
    results = []
    for _ in list_of_args:
        result = rslt_queue.get()
        logger.debug(f"  GOT a result")
        results.append(result)

    end_dt = datetime.now()
    logger.info(f"Completed MP Queue at {end_dt.strftime('%Y-%m-%d %I:%M')} "
                f"with {len(results)} results.")
    for handler in logger.handlers:
        handler.flush()

    return results
