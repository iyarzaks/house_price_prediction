from potentials_creation import read_from_pkl

def main():
    all_single_potentials = read_from_pkl("all_data.pkl")
    couple_potentials = read_from_pkl("potentials_dict.pkl")
    print(len(couple_potentials.keys()))
    print (all_single_potentials.iloc[0])
    print (couple_potentials[(0, 524)])


if __name__ == '__main__':
    main()
