from data.HOL4.utils import add_databases, gen_pretraining_data, gen_databases


def main():
    gen_databases.gen_hol4_data()
    gen_pretraining_data.gen_pretraining_data()
    add_databases.add_databases()

if __name__ == '__main__':
    main()
