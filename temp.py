def f(a):
    len_a = sum(a)
    res = 1
    for i in a:
        res -= (i / len_a) ** 2
    return res


def g(a, b):
    len_a = sum(a)
    len_b = sum(b)
    if len_a == 0:
        return f(b) * len_b / (len_a + len_b)
    elif len_b == 0:
        return f(a) * len_a / (len_a + len_b)
    else:
        return f(a) * len_a / (len_a + len_b) + f(b) * len_b / (len_a + len_b)


inputs = [
    ([1, 0], [4, 3]),
    ([2, 0], [3, 3]),
    ([3, 0], [2, 3]),
    ([3, 1], [2, 2]),
    ([4, 1], [1, 2]),
    ([4, 2], [1, 1]),
    ([5, 2], [0, 1]),
    ([5, 3], [0, 0]),
    ([1, 2], [4, 1]),
]

# print(f([5, 3]))
# print(g([1, 0], [4, 3]))

for i in inputs:
    print(g(*i))
