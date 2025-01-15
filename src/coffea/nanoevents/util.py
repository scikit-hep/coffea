import urllib.parse


def quote(x):
    return urllib.parse.quote(x, safe="")


def unquote(x):
    return urllib.parse.unquote(x)


def concat(*keys):
    return quote(",".join(unquote(k if k is not None else "") for k in keys))


def tuple_to_key(tup):
    return "/".join(quote(x) for x in tup)


def key_to_tuple(key):
    return tuple(unquote(x) for x in key.split("/"))
