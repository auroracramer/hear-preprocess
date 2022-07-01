def opt_list(val, cond):
    """ Conditionally define list elements for appending inline """
    return [val] if cond else []


def opt_tuple(val, cond):
    """ Conditionally define tuple elements for appending inline """
    return (val,) if cond else ()


def opt_dict(key, val, cond):
    """ Conditionally define tuple elements for appending inline """
    return {key: val} if cond else {}