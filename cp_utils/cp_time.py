from datetime import datetime


def datetime_now():
    return datetime.now()


def datetime_log():
    return datetime.now().__str__()[:-7]


def time_log():
    return datetime.now().time().__str__()[:-7]


def datetime_log_fancy() -> str:
    t = datetime_log()
    assert isinstance(t, str)
    t = t.replace(' ', '_')
    t = t.replace(':', '-')
    return t


def time_log_fancy():
    t = time_log()
    assert isinstance(t, str)
    t = t.replace(' ', '_')
    t = t.replace(':', '-')
    return t


def datetime_diff(t1, t2=None, show_mili=False):
    if t2 is None:
        t2 = datetime.now()

    diff = t2 - t1
    diff_min = diff.seconds // 60
    diff_sec = diff.seconds % 60
    diff_micro = diff.microseconds // 1000
    result = f'{diff_min} mins {diff_sec} secs'
    if show_mili or diff_micro > 500:
        result += f' {diff_micro} us'
    return result


def datetime_diff_ms(t1, t2=None):
    if t2 is None:
        t2 = datetime.now()

    diff = t2 - t1
    diff_min = diff.seconds // 60
    diff_sec = diff.seconds % 60
    diff_micro = diff.microseconds // 1000

    total = diff_micro + (diff_sec + diff_min * 60) * 1000
    return total
