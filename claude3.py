import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import decimal
import re
import PyPDF2


#Costanti che ci servono
G = 6.67430e-11    # Costante di gravitazione universale (N m^2/kg^2)
R = 8.3145         # Costante dei gas ideali in J/(mol·K)
R_earth = 6.371e6  # Raggio della Terra (m)
M_earth = 5.972e24 # Massa della Terra (kg)
R_sun = 6.96e8     # Raggio del Sole (m)
M_sun = 1.989e30   # Massa del Sole (kg)
S_B = 5.67e-8      # Costante di Stephan-Boltzmann espressa in W*m^2 / K^4
k_B = 1.3806e-23   # Costante di Boltzmann (J/K)
Pr_t = 0.9         # Numero di Prandtl turbolento (costante empirica)

# Legge ed estrae i parametri dal file di configurazione
def read_config(config_file):
    params = {}
    with open(config_file, 'r') as f:
        for line in f:
            if '#' in line:
                # Rimuovi il commento dalla riga
                line = line[:line.index('#')]
            if '=' in line:
                key, value = line.strip().split('=')
                params[key.strip()] = float(value.strip().split()[0])
    return params

config = read_config('config.txt')
time = config['time']
delta_t = config ['delta_t']
Stellar_T = config['Stellar-T']
Stellar_R = config['Stellar-R']
Stellar_M = config['Stellar-M']
Planet_R = config['Planet-R']
Planet_M = config['Planet-M']
Planet_distance = config['Planet_distance']
scale_height = config['scala_altitudine']
c_p = config['c_p']
Stellar_L = config['Stellar-L']
Albedo = config['Albedo']
P_0 = config['P_0']



# Scelta de file contenente i mixing ratio delle specie chimiche
# a cui applicare il vertical mixing

dat_files = glob.glob('*.dat')

print("Risultati disponibili:\n")
for i, filename in enumerate(dat_files, 1):
    print(f"{i}. {filename}")

file_index = int(input("\nSeleziona il file da elaborare: ")) - 1
selected_file = dat_files[file_index]
atmo_data = np.loadtxt(selected_file)

# funzione che legge le specie chimiche e mette in una tupla 
# tutti i parametri delle specie chimiche elencate, nomi, pesi e epsilon

def read_species_list(file_path):
    species_list = {}
    scientific_pattern = r'^(\d+(\.\d+)?([eE][-+]?\d+)?)'
    default_epsilon = decimal.Decimal('1.04e-17')  # valore di epsilon di default

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                fields = line.split('-')
                species_name = fields[0].strip()
                species_mass = float(fields[1].strip())
                species_epsilon = default_epsilon  # assegna il valore di default
                if len(fields) > 2 and fields[2].strip():
                    epsilon_str = fields[2].strip()
                    match = re.match(scientific_pattern, epsilon_str)
                    if match:
                        try:
                            species_epsilon = decimal.Decimal(match.group(1))
                        except decimal.InvalidOperation:
                            print(f"Avviso: valore di epsilon non valido per {species_name}")
                    else:
                        print(f"Avviso: valore di epsilon non valido per {species_name}")
                species_list[species_name] = (species_mass, species_epsilon)
    return species_list

# chiede quanti frames elaborare
num_frames = int(input("Inserisci il numero di frames da elaborare: ")) -1

# Determina il numero di strati atmosferici dal file di input
delta_z = atmo_data.shape[0]

# Crea un array di quote atmosferiche
z = np.arange(0, scale_height, delta_z)[:atmo_data.shape[0]]

# Calcolo del flusso radiante ricevuto dal pianeta
stellar_flux = (1 - Albedo) * Stellar_L / (4 * np.pi * Planet_distance**2)

# funzione che calcola i pesi molecolari medi di ogni strato atmosferico
# a partire dai pesi molecolari in species e restituisce un array di mu 
def calculate_mean_molecular_weight(atmo_data, species_list):
    num_layers = atmo_data.shape[0]
    num_species = atmo_data.shape[1]
    mu = np.zeros(num_layers)

    # Ottieni i nomi delle specie dal dizionario species_list
    species_names = list(species_list.keys())

    for i in range(num_layers):
        total_mixing_ratio = np.sum(atmo_data[i, :])
        if total_mixing_ratio > 0:
            normalized_atmo_data = atmo_data[i, :] / total_mixing_ratio
            layer_mu = 0
            for j, species_name in enumerate(species_names):
                if j < num_species:
                    species_mass, _ = species_list.get(species_name, (0.0, 0.0))
                    layer_mu += normalized_atmo_data[j] * species_mass
            mu[i] = layer_mu
        else:
            # Se la somma dei mixing ratio è 0, imposta il peso molecolare medio a 0
            mu[i] = 0
    return mu

# la funzione che salva i mu medi per ogni strato atmosferico in un file
def save_mu_profile(mu, file_path):
    np.savetxt(file_path, mu, fmt='%.6f')

# questa funzione calcola i coefficienti estinzione molare medi per ogni strato atmosferico
# è l'equivalente di quanto fatto per i mu medi, serve a stimare l'opacità radiativa di 
# ciascuno strato e quindi calcolare i flussi radiativi entranti e uscenti

def calculate_mean_epsilon(atmo_data, species_list):
    num_layers = atmo_data.shape[0]
    num_species = atmo_data.shape[1]
    epsilon = np.zeros(num_layers)

    species_names = list(species_list.keys())

    for i in range(num_layers):
        total_mixing_ratio = np.sum(atmo_data[i, :])
        if total_mixing_ratio > 0:
            normalized_atmo_data = atmo_data[i, :] / total_mixing_ratio
            layer_epsilon = decimal.Decimal('0.0')
            for j, species_name in enumerate(species_names):
                if j < num_species:
                    species_mass, species_epsilon = species_list.get(species_name, (0.0, decimal.Decimal('0.0')))
                    layer_epsilon += decimal.Decimal(str(normalized_atmo_data[j])) * species_epsilon
            epsilon[i] = float(layer_epsilon)
        else:
            epsilon[i] = 0.0

    return epsilon

#calcolo la variazione di gravità in funzione della quota
def get_local_gravity(z):
    r_local = Planet_R * R_earth + z
    g_local = G * Planet_M * M_earth / r_local**2
    return g_local

# la funzione che calcola il gradiente di viscosità cinematica nu
def calculate_kinematic_viscosity(T, P):
    nu = k_B * T / (2.6696e-26 * P)
    return nu

# la funzione che calcola il coefficiente di diffusione in funzione di nu
def calculate_diffusion_coefficient(nu):
    k_v = nu / Pr_t
    return k_v

    
def get_surface_temperature(P_0, Planet_M, Planet_R, Stellar_T, Stellar_R, Planet_distance, Albedo, mu, g_local, tol=1e-3, max_iter=100):
    """
    Calcola la temperatura del suolo (T0) in modo iterativo, utilizzando i parametri del pianeta e dell'atmosfera.

    Parametri:
    P_0 (float): pressione a livello del suolo in bar
    Planet_M (float): massa del pianeta in masse terrestri
    Planet_R (float): raggio del pianeta in raggi terrestri
    Stellar_T (float): temperatura della stella in Kelvin
    Stellar_R (float): raggio della stella in raggi solari
    Planet_distance (float): distanza media pianeta-stella in metri
    Albedo (float): albedo del pianeta
    mu (np.ndarray): profilo di peso molecolare medio
    g_local[0] (float): accelerazione di gravità a livello del suolo
    tol (float): tolleranza per la convergenza (opzionale, default=1e-3)
    max_iter (int): numero massimo di iterazioni (opzionale, default=100)

    Restituisce:
    T0 (float): temperatura del suolo in Kelvin
    """
    # Inizializza T0 utilizzando l'equazione di equilibrio radiativo per un pianeta senza atmosfera
    T0 = Stellar_T * np.sqrt(Stellar_R / (2 * Planet_distance)) * np.sqrt(1 - Albedo)

    # Calcola rho_0 e il profilo di densità rho utilizzando T0 iniziale
    rho_0 = P_0 * 1e5 / (R * T0)  # Converte la pressione da bar a Pa
    rho = get_density_profile(z, mu, g_local, T_profile=None)

    # Itera finché T0 non converge
    for i in range(max_iter):
        T0_old = T0
        rho_0 = rho[[1]]  # Densità a livello del suolo
        H = R * rho_0 / (P_0 * 1e5 * mu[[1]])  # Altezza di scala barometrica
        T0 = P_0 * 1e5 / (rho_0 * R / H)  # Temperatura del suolo utilizzando l'equazione dell'atmosfera isoterma

        # Aggiorna il profilo di densità rho con la nuova T0
        rho = get_density_profile(z, mu, g_local, T_profile=None)

        # Verifica la convergenza
        if abs(T0 - T0_old) < tol:
            break
    return T0

    
# trova il gradiente di temperatura più sofisticato
def get_temperature_profile(z, Stellar_T, Stellar_R, Planet_distance, c_p, rho, epsilon):
    """
    Parametri:
    z (np.ndarray): profondità (m)
    Stellar_T (float): temperatura della stella (K)
    Stellar_R (float): raggio della stella (in raggi solari)
    Stellar_M (float): massa della stella (in masse solari)
    Planet_distance (float): distanza media pianeta-stella (km)
    c_p (float): calore specifico a pressione costante (J/kg/K)
    rho (np.ndarray): profilo di densità (kg/m^3)
    epsilon (np.ndarray): profilo di epsilon medio
    Restituisce:
    T (np.ndarray): profilo di temperatura (K)
    """
    # Inizializza il profilo di temperatura
    T = np.zeros_like(z[:atmo_data.shape[0]])
    T[[1]] = T0

    # Costanti utili
    g = G * Planet_M * M_earth / (Planet_R * R_earth)**2
    dt = delta_t
    dz = delta_z

    # Iterazione implicita per il profilo di temperatura
    for i in range(len(z)):
        kappa = epsilon[i] * rho[i] * dz
        F_out = kappa * S_B * T[i-1]**4
        F_net = stellar_flux - F_out

        # Metodo implicito di Eulero
        T[i] = T[i-1] + dt * (
            - g / (c_p * mu[i] * rho[i])
            - F_net / (c_p * rho[i] * dz)
        )

    return T

# calcola il profilo di pressione    
def get_pressure_profile(z, rho, T_profile, mu, P_0):
    n = len(z)
    P = np.zeros(n)
    P[0] = P_0

    for i in range(1, n):
        r = Planet_R * R_earth + z[i]
        g_local = G * Planet_M * M_earth / r**2
        dp_dz = -rho[i] * g_local
        dz = z[i] - z[i-1]

        # Integrazione per ottenere la pressione all'altezza z[i]
        if i == n - 1:
            # Nell'ultimo strato, imposta la pressione a zero
            P[i] = 0
        else:
            P[i] = P[i-1] + dp_dz * dz

        # Calcola la pressione usando l'equazione di stato dei gas ideali
        P[i] = rho[i] * R * T_profile[i] / mu[i]
    return P

    
def get_density_profile(z, mu, g_local, T_profile=None):
    """
    Calcola il profilo di densità atmosferica in funzione dell'altezza.
    
    Parametri:
    z (array): Altezze [m] per le quali calcolare il profilo di densità.
    T_profile (array, opzionale): Profilo di temperatura [K] in funzione di z.
                                 Se non fornito, viene assunto un valore di equilibrio.
    rho_0 (float, opzionale): Densità a livello del suolo [kg/m^3].
                             Se non fornito, viene calcolato.
    mu (float): Massa molare media dell'atmosfera [kg/mol].
    g_local (array): Accelerazione di gravità locale [m/s^2] in funzione di z.
                     
    Restituisce:
    rho (array): Profilo di densità [kg/m^3] in funzione di z.
    """
    # Calcola la temperatura a livello del suolo se non fornita
    if T_profile is None:
        T_0 = Stellar_T * np.sqrt(Stellar_R / (2 * Planet_distance)) * np.sqrt(1 - Albedo)
    else:
        T_0 = T_profile[0]

    rho_0 = P_0 / (R * T_0)
    H = R * Stellar_T / (mu * g_local)

    if T_profile is None:
        rho = rho_0 * np.exp(-z[:len(g_local)] / H[:len(g_local)])
    else:
        rho = rho_0 * np.exp(-np.cumsum(dz) / H)

    # Applicare un raccordo esponenziale per i valori di densità vicino all'interfaccia con lo spazio
    z_top = z[-5]
    rho_top = rho[-6]
    for i in range(-5, 0):
        rho[i] = rho_top * np.exp(-(z[i] - z_top) / (scale_height / 5))

    return rho

#Calcola i flussi verticali e applica l'algoritmo di vertical mixing ai dati atmosferici ricorsimavamente
def vertical_mixing_recursive(atmo_data, z, k_v, delta_z, delta_t, time, rho, T_profile):
    """
    Parametri:
    atmo_data (np.ndarray): dati atmosferici (mixing ratio delle specie chimiche)
    z (np.ndarray): la lista delle varie quote dei vari strati atmosferici (m)
    k_v (float): coefficiente di diffusione turbolenta verticale (m^2/s)
    delta_z (float): incremento altitudine (m)
    delta_t (float): intervallo di tempo (s)
    time (float): tempo finale in (s)
    rho (np.ndarray): profilo di densità (kg/m^3)
    T_profile (np.ndarray): profilo di temperatura (K)
    
    Restituisce:
    atmo_data_mixed (np.ndarray): dati atmosferici dopo il mixing (mixing ratio delle specie chimiche)
    """
    
    # Inizializza i dati atmosferici 
    atmo_data_mixed = atmo_data.copy()
    temperature_profile = T_profile.copy()
    vertical_fluxes = np.zeros_like(atmo_data)
    num_iterations = int(time / delta_t)

    def update_concentrations(atmo_data_mixed, iteration):
        temp_atmo_data_mixed = atmo_data_mixed.copy()
        for i in range(1, len(z)):
            for j in range(atmo_data.shape[1]):
                delta_C = atmo_data_mixed[i, j] - atmo_data_mixed[i - 1, j]
                F = -k_v[i] * (delta_C / delta_z + rho[i] * T_profile[i] / rho[i - 1] / T_profile[i - 1] * delta_C / delta_z)
                temp_atmo_data_mixed[i, j] = atmo_data_mixed[i, j] + F * delta_t / delta_z
                vertical_fluxes[i, j] = F
        if iteration % (num_iterations // 10) == 0:
            progress = ((iteration + 1) / num_iterations) * 100
            print(f"{progress:.0f}% completato")
        return temp_atmo_data_mixed

    for iteration in range(1, num_iterations + 1):
        atmo_data_mixed = update_concentrations(atmo_data_mixed, iteration)

    np.savetxt('temperature.txt', temperature_profile)
    np.savetxt('vertical_flux.txt', vertical_fluxes)
    return atmo_data_mixed
 
# Calcola il mixing convettivo nell'atmosfera
def convective_mixing(atmo_data, z, T_profile, mu, rho, c_p, g_local, delta_t, time):
    """
    Parametri:
    atmo_data (np.ndarray): dati atmosferici (mixing ratio delle specie chimiche)
    z (np.ndarray): quote atmosferiche (m)
    T_profile (np.ndarray): profilo di temperatura (K)
    mu (np.ndarray): profilo di peso molecolare medio (kg/mol)
    rho (np.ndarray): profilo di densità (kg/m^3)
    c_p (float): calore specifico a pressione costante (J/kg/K)
    g_local (np.ndarray): profilo di accelerazione di gravità locale (m/s^2)
    delta_t (float): intervallo di tempo (s)
    time (float): tempo finale in (s)
    
    Restituisce:
    atmo_data_convected (np.ndarray): dati atmosferici dopo il mixing convettivo
    """
    gamma = mu * g_local / (R * c_p)
    atmo_data_convected = atmo_data.copy()
    num_iterations = int(time / delta_t)
    convective_displacements = np.zeros_like(atmo_data)

    def update_convection(atmo_data_convected, i):
        dT_dz = (T_profile[i] - T_profile[i - 1]) / (z[i] - z[i - 1])

        if dT_dz < -gamma[i]:
            atmo_data_convected[i] = atmo_data_convected[i - 1]
            convective_displacements[i] = atmo_data_convected[i] - atmo_data[i]
        elif dT_dz > -gamma[i]:
            pass
        else:
            pass

        if i < len(z) - 1:
            atmo_data_convected = update_convection(atmo_data_convected, i + 1)

        return atmo_data_convected

    for iteration in range(num_iterations):
        atmo_data_convected = update_convection(atmo_data_convected, 1)

    np.savetxt('convective_displacements.txt', convective_displacements)
    return atmo_data_convected

# la funzione che reitera l'elaborazione tante volte quante il numero di frames digitato dall'utente
def process_frames(atmo_data, num_frames, selected_file, z, k_v, delta_z, delta_t, time, rho, T_profile):
    filename = selected_file
    for i in range(num_frames):
        atmo_data = np.loadtxt(filename)
        atmo_data_mixed = vertical_mixing_recursive(atmo_data, z, k_v, delta_z, delta_t, time, rho, T_profile)
        output_filename = f"atmo_mixed{i+1}.dat"
        np.savetxt(output_filename, atmo_data_mixed)
        print(f"Frame {i+1} di {num_frames+1} completati")
        filename = output_filename

# Leggi la lista delle specie chimiche dal file species
species_list = read_species_list('species')

# Calcola il gradiente di gravità
g_local = get_local_gravity(z)

# Calcola il profilo di peso molecolare medio
mu = calculate_mean_molecular_weight(atmo_data, species_list)

# Calcola il profilo di epsilon medio
epsilon = calculate_mean_epsilon (atmo_data, species_list)

# Calcola il profilo di densità con un algoritmo che utilizza solo peso molecolare, altitudine e gravità
rho = get_density_profile(z, mu, g_local) 

# calcola la temperatura a livello del suolo
T0 = get_surface_temperature(P_0, Planet_M, Planet_R, Stellar_T, Stellar_R, Planet_distance, Albedo, mu, g_local)

# Calcola il profilo di temperatura usando il profilo di densità semplificato
T_profile = get_temperature_profile(z, Stellar_T, Stellar_R, Planet_distance, c_p, rho, epsilon)

# Calcola il profilo di densità con un algoritmo che utilizza anche la temperatura
rho = get_density_profile(z, mu, g_local, T_profile=None)

# Calcola il profilo di pressione
P = get_pressure_profile(z, rho, T_profile, mu, P_0 * 1e5)

# Rielabora il profilo di temperatura 
T_profile = get_temperature_profile(z, Stellar_T, Stellar_R, Planet_distance, c_p, rho, epsilon)

# Calcola il profilo di densità con un algoritmo che utilizza anche la temperatura
rho = get_density_profile(z, mu, g_local, T_profile=None)

# Calcolo della viscosità cinematica e del coefficiente di diffusione
nu = calculate_kinematic_viscosity(T_profile, P)
k_v = calculate_diffusion_coefficient(nu)

# Salva il profilo di peso molecolare mu, temperatura e flussi verticali
save_mu_profile(mu, 'mu.txt')

# Salva il profilo di densità
np.savetxt('density.txt', rho)

# Salva il profilo di pressione
np.savetxt('pressure.txt', P)

# Salva il profilo di viscosità cinematica
np.savetxt('kinematic_viscosity.txt', nu)

# Salva il profilo del coefficiente di estinzione molare epsilon
np.savetxt('epsilon.txt', epsilon)

# Salva il profilo del coefficiente di diffusione k_v
np.savetxt('diffusion_coefficient.txt', k_v)

# avvia il ciclo di elaborazione a molti frames
process_frames(atmo_data, num_frames, selected_file, z, k_v, delta_z, delta_t, time, rho, T_profile)

# Applica i moti convettivi 
atmo_data_mixed_turb = vertical_mixing_recursive(atmo_data, z, k_v, delta_z, delta_t, time, rho, T_profile)

# Calcola il mixing convettivo dovuto all'instabilità statica
atmo_data_mixed_conv = convective_mixing(atmo_data, z, T_profile, mu, rho, c_p, g_local, delta_t, time)

# Inizializza l'array per i dati atmosferici miscelati finali
atmo_data_mixed = atmo_data.copy()

# Combina i risultati dei due processi utilizzando la somma dei coefficienti di mixing
for i in range(len(z)):
   for j in range(atmo_data.shape[1]):
       atmo_data_mixed[i, j] = atmo_data_mixed_turb[i, j] + atmo_data_mixed_conv[i, j]

# Da qui in poi i codici per plottare i vari gradienti
# dati per l'asse delle altezze
altitude_increment = scale_height / delta_z  # Incremento di altitudine per layer in metri
z_altitude = z * altitude_increment
# Converti l'altitudine da metri a chilometri
z_altitude_km = z_altitude / 100000

# Carica i dati dai file di testo
mu_profile = np.loadtxt('mu.txt')
temp_profile = np.loadtxt('temperature.txt')
vertical_fluxes = np.loadtxt('vertical_flux.txt')
density_profile = np.loadtxt('density.txt')
pressure_profile = np.loadtxt('pressure.txt')
kinematic_viscosity_profile = np.loadtxt('kinematic_viscosity.txt')
epsilon_profile = np.loadtxt('epsilon.txt')
diffusion_coefficient_profile = np.loadtxt('diffusion_coefficient.txt')
convective_displacements = np.loadtxt('convective_displacements.txt')

# Grafico di mu
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(mu_profile, z_altitude_km)
ax.set_xlabel('Mean molecular weight (kg/mol)')
ax.set_ylabel('Altitude (km)')
ax.set_title(f'Mu profile - {selected_file}')
ax.yaxis.set_ticks(np.arange(0, np.max(z_altitude_km), 10))
ax.grid(True)
plt.savefig('mu_profile.pdf')

# Grafico Temperature
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(temp_profile, z_altitude_km)
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Altitude (km)')
ax.set_title(f'Temperature profile - {selected_file}')
ax.yaxis.set_ticks(np.arange(0, np.max(z_altitude_km), 10))
ax.grid(True)
plt.savefig('temp_profile.pdf')

# Crea il grafico del profilo di flusso verticale totale
total_vertical_flux = np.sum(vertical_fluxes, axis=1) #Calcola il flusso verticale totale per ogni strato
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(total_vertical_flux, z_altitude_km)
ax.set_xlabel('Total vertical flux (kg/m^2/s)')
ax.set_ylabel('Altitude (km)')
ax.set_title(f'Total vertical flux profile - {selected_file}')
ax.yaxis.set_ticks(np.arange(0, np.max(z_altitude_km), 10))
ax.grid(True)
plt.savefig('total_vertical_flux.pdf')

# Crea il grafico del profilo di spostamento convettivo
convective_displacements = np.sum(convective_displacements, axis=1)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(convective_displacements, z_altitude_km)
ax.set_xlabel('delta Mixing Ratio (%)')
ax.set_ylabel('Altitude (km)')
ax.set_title(f'Moti convettivi totali - {selected_file}')
ax.yaxis.set_ticks(np.arange(0, np.max(z_altitude_km), 10))
ax.grid(True)
plt.savefig('convective_displacements.pdf')

# Grafico della densità
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(density_profile, z_altitude_km)
ax.set_xlabel('Densità (kg/m³)')
ax.set_ylabel('Altitude (km)')
ax.set_title(f'Profilo di densità - {selected_file}')
ax.yaxis.set_ticks(np.arange(0, np.max(z_altitude_km), 10))
ax.grid(True)
plt.savefig('density_profile.pdf')

# Grafico della pressione
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(P/101325, z_altitude_km)   #converti in atm
ax.set_xlabel('Pressione (atm)')
ax.set_ylabel('Altitude (km)')
ax.set_title(f'Profilo di pressione - {selected_file}')
ax.yaxis.set_ticks(np.arange(0, np.max(z_altitude_km), 10))
ax.grid(True)
plt.savefig('pressure_profile.pdf')

# Grafico della viscosità
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(nu, z_altitude_km)
ax.set_xlabel('Viscosità cinematica (m²/s)')
ax.set_ylabel('Altitude (km)')
ax.set_title(f'Profilo di viscosità cinematica - {selected_file}')
ax.yaxis.set_ticks(np.arange(0, np.max(z_altitude_km), 10))
ax.grid(True)
plt.savefig('kinematic_viscosity_profile.pdf')

# Grafico del coefficiente di estinzione molare epsilon
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(epsilon, z_altitude_km)
ax.set_xlabel('Coefficiente di estinzione molare ε (m²/mol)')
ax.set_ylabel('Altitude (km)')
ax.set_title(f'Profilo del coefficiente di estinzione molare - {selected_file}')
ax.yaxis.set_ticks(np.arange(0, np.max(z_altitude_km), 10))
ax.grid(True)
plt.savefig('epsilon_profile.pdf')

# Grafico del coefficiente di diffusione k_v
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(k_v, z_altitude_km)
ax.set_xlabel('Coefficiente di diffusione (m²/s)')
ax.set_ylabel('Altitude (km)')
ax.set_title(f'Profilo del coefficiente di diffusione - {selected_file}')
ax.yaxis.set_ticks(np.arange(0, np.max(z_altitude_km), 10))
ax.grid(True)
plt.savefig('diffusion_coefficient_profile.pdf')

#cancella tutti i files dei dati generati
os.remove('mu.txt')
os.remove('temperature.txt')
os.remove('vertical_flux.txt')
os.remove('density.txt')
os.remove('pressure.txt')
os.remove('kinematic_viscosity.txt')
os.remove('epsilon.txt')
os.remove('diffusion_coefficient.txt')
os.remove('convective_displacements.txt')


# Funzione per unire tutti i grafici in un unico pdf
def merge_pdf_files(pdf_files, output_filename):
    pdf_merger = PyPDF2.PdfMerger()

    for pdf_file in pdf_files:
        with open(pdf_file, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            pdf_merger.append(pdf_reader)

    with open(output_filename, 'wb') as output_file:
        pdf_merger.write(output_file)

# Lista dei file PDF da unire
# Lista dei file PDF da unire
pdf_files = ['pressure_profile.pdf', 'temp_profile.pdf', 'density_profile.pdf', 'mu_profile.pdf', 'epsilon_profile.pdf', 'diffusion_coefficient_profile.pdf', 'kinematic_viscosity_profile.pdf', 'total_vertical_flux.pdf', 'convective_displacements.pdf']

# Unisci
merge_pdf_files(pdf_files, selected_file + ".pdf")

# Cancella tutti i grafici singoli
for pdf_file in pdf_files:
    os.remove(pdf_file)

