# ASyH concurrent/async dispatch
# ToDos: -

from multiprocessing import Process, Value


def task(pipeline, retval):
    retval.value = pipeline.run()


def concurrent_dispatch(*pipelines):
    '''Run several ASyH pipelines concurrently with multiprocessing.Process.'''
    results = []
    procs = []

    for i in range(len(pipelines)):
        ret = Value('f', 0.0)
        p = Process(target=task, args=(pipelines[i], ret))
        results.append(ret)
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    return [results[i].value for i in range(len(pipelines))]
