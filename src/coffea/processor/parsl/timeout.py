from functools import wraps


def timeout(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import signal

        def _timeout_handler(signum, frame):
            raise Exception("Timeout hit")

        signal.signal(signal.SIGALRM, _timeout_handler)
        if kwargs.get("timeout"):
            signal.alarm(max(1, int(kwargs["timeout"])))
        try:
            result = func(*args, **kwargs)
        finally:
            signal.alarm(0)
        return result

    return wrapper
