# ASyH concurrent/async dispatch
# ToDos:
#   Joining and collecting return values.
#
#   Module docstring

import asyncio


def concurrent_dispatch(*pipelines):
    '''Run several ASyH pipelines concurrently with asyncio.'''
    for p in pipelines:
        asyncio.run(p.p_run())
