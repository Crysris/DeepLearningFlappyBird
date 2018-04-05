from numpy import *

q = zeros((6, 6))
rewards = zeros((6, 6))
rewards[:, 5] = 500
actions = [[4], [3, 5], [3], [1, 2, 4], [0, 3, 5], [1, 4, 5]]


def trial():
    s = random.randint(6)
    while s < 5:
        s1 = a = random.choice(actions[s])
        q[s, a] = rewards[s, a] + 0.8 * q[s1].max()
        s = s1


print(q)
for i in range(100):
    trial()

print(q)


def test(s):
    print(s),
    while s < 5:
        s = q[s].argmax()
        print(
            "->",
            s,
        )


test(2)
