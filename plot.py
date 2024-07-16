import matplotlib.pyplot as plt
import numpy as np
import glob


# Inizializzazione del dizionario
species = {}

# Apertura del file in modalità lettura
with open('species', 'r') as file:
    # Lettura di ogni riga del file
    for i, line in enumerate(file):
        # Rimozione degli spazi bianchi iniziali e finali
        line = line.strip()
        # Estrazione del nome della specie chimica
        name = line.split()[0]
        # Aggiunta della specie al dizionario con il suo numero di riga come valore
        species[name.lower()] = i + 1



dat_files = glob.glob('*.dat')

# Print the list of .dat files
print("Risultati disponibili:\n")
for i, filename in enumerate(dat_files, 1):
    print(f"{i}. {filename}")

# Ask the user to select a file
file_index = int(input("\nInserisci l'indice del file da analizzare: ")) - 1
selected_file = dat_files[file_index]

# Load data
data = np.loadtxt(selected_file)

# Find species with values greater than 1e-14 and their max values
species_to_plot = [(name, np.max(data[:, column-1])) for name, column in species.items() if column-1 < data.shape[1] and np.any(data[:, column-1] > 1e-14)]

# Print report
print("\nSpecie chimiche che hanno superato un mix ratio di 10^-14:\n")
for i, name in enumerate(species_to_plot, 1):
    print(f"{i}. {name}")

# Ask user for input
plot_all = input("\nVuoi mostrarle tutte? (s/n) ")
if plot_all.lower() != 's':
    indices_to_plot = input("Seleziona le specie. Singolarmente separate da ','  o con '-' per un intervallo: ")
    indices_to_plot = indices_to_plot.split(',')
    expanded_indices = []
    for index in indices_to_plot:
        if '-' in index:
            start, end = map(int, index.split('-'))
            expanded_indices.extend(range(start, end + 1))
        else:
            expanded_indices.append(int(index))
    indices_to_plot = [index - 1 for index in expanded_indices]
    species_to_plot = [species_to_plot[index] for index in indices_to_plot]

# Creazione del grafico
plt.figure(figsize=(10, 10)) # dimensioni del grafico
plt.rcParams['axes.facecolor'] = 'white' # colore di sfondo degli assi
plt.grid(color='gray', linestyle='--', linewidth=0.5) # griglia
plt.yscale('log')
plt.xscale('log')
plt.xlim([1e-14, 1])
plt.ylim([1, 1e-10])
plt.xlabel('Mixing ratio')
plt.ylabel('Pressure (bar)')
plt.title(selected_file)

for name, _ in species_to_plot:
    plt.plot(data[:, species[name]-1], data[:, 1], label=name)

# Mostra la legenda e salva il grafico in un file PDF
plt.legend()
# Salva il grafico in formato PNG invece di PDF
plt.savefig("plot.png", dpi=300, bbox_inches='tight')
plt.show()
