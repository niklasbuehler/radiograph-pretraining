import time
_print_ratelimited = (None, 0, 0)

def print_ratelimited(*values,
                      sep = " ",
                      end = "\n",
                      file=None,
                      flush=True,
                      reset_timedelta_seconds=5):
    global _print_ratelimited
    prevstr, prevtime, prevoccurrences = _print_ratelimited
    curstr = sep.join(str(obj) for obj in values)
    curtime = time.time()
    if curstr != prevstr or curtime - prevtime >= reset_timedelta_seconds:
        prevoccurrences = 0
    if prevoccurrences == 0:
        print(curstr, end=end, file=file, flush=flush)
    elif prevoccurrences == 1:
        print('[RATELIMITING]', curstr, end=end, file=file, flush=flush)
    _print_ratelimited = curstr, curtime, prevoccurrences + 1

def ensureunique(lst):
    assert len(lst) == len(set(lst)), f'{len(lst)=} != {len(set(lst))=}'
    return lst