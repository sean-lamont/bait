from data.HOL4.utils import add_databases, gen_pretraining_data, gen_databases


def main():
    data_dir = 'data/HOL4/data'
    gen_databases.gen_hol4_data(data_dir)
    gen_pretraining_data.gen_pretraining_data(data_dir)
    add_databases.add_databases()

if __name__ == '__main__':
    main()
