import json
import os


def gen_hol4_data():

    data_dir = "data/hol4/data/"
    data_dir = os.path.join(os.getcwd(),data_dir)

    with open(data_dir + "hol_data.txt") as fp:
        x = fp.readlines()

    y = "".join(x)

    y = y.replace("\n", "")

    y = y.replace("  ", " ")
    y = y.replace("  ", " ")

    z = y.split("|||")

    ret = []
    buf = []
    j = 1
    i = 0
    while i < len(z):
        # at end of entry
        cur = z[i]
        if j % 6 == 0:
            buf.append(cur)
            j = 1
            ret.append(buf)
            buf = []
        elif cur == 'thm':
            buf.append(cur)
        else:
            buf.append(cur)
            j += 1
        i += 1

    # new database mapping from theory-number to values (much smaller key than polished goal)

    # 1. mapping theory to dependencies (e.g. list-25 : [list-24, bool-23, bool-2])
    # 2. mapping theory/def name to values

    dep_dict = {}
    db_dict = {}

    for term in ret:
        # if thm
        if len(term) == 7:
            dep_dict[str(term[0]) + "-" + str(term[3])] = term[5].split(", ")

        db_dict[str(term[0]) + "-" + str(term[3])] = [term[0], term[1], term[2][2:], term[3], term[4], term[-1][2:]]

    # remove leading whitespace

    full_db = {}
    for key in db_dict.keys():
        val = db_dict[key]

        if key[0] == " ":
            full_db[key[1:]] = val
        else:
            full_db[key] = val

    deps = {}
    for key in dep_dict.keys():
        val = dep_dict[key]

        if key[0] == " ":
            deps[key[1:]] = val
        else:
            deps[key] = val

    #dependency database for theorems
    with open(data_dir + "dep_data.json", "w") as f:
        json.dump(deps, f)

    #database for all expressions and their info (e.g. thm vs dep, dep number, utf encoding)
    with open(data_dir + "new_db.json", "w") as f:
        json.dump(full_db, f)

if __name__ == '__main__':
    gen_hol4_data()
