
a = [1, 2, 4, 5, 7, 3]
b = [2, 8, 9, 7, 10]

cinos = list(set(a) & set(b))
idx2 = [b.index(ino) for ino in cinos]

c = a.copy()
for ino in b:
    ### Do nothing already exists ###
    if ino in a:
        pass
    ### Does not exist, is it exess or disjoint? ###
    else:
        if max(idx2) < b.index(ino):
            print("Excess: ", ino)
        else:
            for cino, idx in zip(cinos, idx2):
                if b.index(ino) < idx:
                    print("disjoint", ino)
                    c.insert(c.index(cino), ino)

print(a)
print(b)
print(c)
# print(idxs)
