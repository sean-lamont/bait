from data.hol4 import gen_databases, gen_pretraining_data, add_databases

def main():
    gen_databases.gen_hol4_data()
    gen_pretraining_data.gen_pretraining_data()
    add_databases.add_databases()

if __name__ == '__main__':
    main()
