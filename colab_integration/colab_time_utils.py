#@title COLAB Time utils

from datetime import datetime


def t_now():
    return datetime.now()


def t_log():
    return datetime.now().__str__()[:-7]


def t_diff(t1, t2=None):
    if t2 is None:
        t2 = datetime.now()

    diff = t2 - t1
    diff_min = diff.seconds // 60
    diff_sec = diff.seconds % 60
    return f'{diff_min} minutes {diff_sec} seconds'
