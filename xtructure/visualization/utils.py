ATOM_SYMBOLS = [
    'H', 'He', 'Li', 'Be', 'B',
    'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P',
    'S', 'Cl', 'Ar', 'K', 'Ca'
]
def write_xyz(filename, atom_numbers, coordinates):
    with open(filename, 'w') as f:
        f.write(str(len(atom_numbers)) + '\n')
        f.write('\n')
        for atom_number, coordinate in zip(atom_numbers, coordinates):
            f.write(ATOM_SYMBOLS[atom_number] + ' ' + ' '.join(map(str, coordinate)) + '\n')
