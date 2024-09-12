# Task 1
import numpy as np
import matplotlib.pyplot as plt

# Constants for the masses (in GeV/c^2)
m_dmeson = 1.97
m_kaon = 0.498
m_pion = 0.135

# Define the function to generate Dalitz plot events with kinematic constraints
def generate_dalitz_events(n, m_d, m_a, m_b, m_c):
    """
    Generate n decay events of D_s+ -> K+ K- pi+ uniformly distributed over the allowed phase space,
    considering the kinematic constraints from the PDG.
    """
    # Empty lists to store the invariant mass squared of the two-particle systems
    m2ab_list = []
    m2bc_list = []
    
    # Loop to generate events
    for _ in range(n):
        # Sample m2ab and m2bc uniformly within the full range of the Dalitz plot
        m2ab = np.random.uniform((m_a + m_b)**2, (m_d - m_c)**2)
        m2bc = np.random.uniform((m_b + m_c)**2, (m_d - m_a)**2)
        
        # Calculate the kinematic limits for m2bc using the PDG equations
        E_c_star = (m_d**2 + m_c**2 - m2ab) / (2*np.sqrt(m2ab))
        p_c_star = np.sqrt(E_c_star**2 - m_c**2)
        
        E_a_star = (m2ab + m_a**2 - m_b**2) / (2*np.sqrt(m2ab))
        p_a_star = np.sqrt(E_a_star**2 - m_a**2)
        
        # Maximum and minimum m2bc values allowed by kinematics
        m2bc_max = (E_c_star + E_a_star)**2 - (p_c_star - p_a_star)**2
        m2bc_min = (E_c_star + E_a_star)**2 - (p_c_star + p_a_star)**2
        
        if m2bc_min < m2bc < m2bc_max:
            # Store the values
            m2ab_list.append(m2ab)
            m2bc_list.append(m2bc)
    
    return m2ab_list, m2bc_list

# Adjust the plotting function to use plt.hist2d with 300x300 bins
def plot_dalitz_hist2d(m2ab_list, m2bc_list, bins=300):
    """
    Plot the Dalitz plot for the generated events using a 2D histogram with specified bins.
    """
    plt.figure(figsize=(8, 6))
    plt.hist2d(m2ab_list, m2bc_list, bins=bins, cmap='viridis')
    plt.colorbar()
    plt.xlabel(r'$m^2(K^+ K^-)$ (GeV$^2/c^4$)')
    plt.ylabel(r'$m^2(K^- \pi^+)$ (GeV$^2/c^4$)')
    plt.title('Dalitz Plot for $D_s^+ \\rightarrow K^+ K^- \pi^+$ Decay')
    plt.show()

# Generate a specified number of events
n_events = 100000

# Use the adjusted function to generate events
m2ab_data, m2bc_data = generate_dalitz_events(n_events, m_dmeson, m_kaon, m_kaon, m_pion)

# Plot the Dalitz plot using hist2d with 300x300 bins
plot_dalitz_hist2d(m2ab_data, m2bc_data, bins=300)

# Task 2

# Mass of the resonance (e.g., for the phi meson) in GeV/c^2
m_res = 1.3  # GeV/c^2

# Decay width of the resonance (e.g., for the phi meson) in GeV
Gamma_res = 0.015  # GeV

# Correct implementation of the Breit-Wigner function and the von Neumann rejection method
# Breit-Wigner probability density function (squared amplitude)
def breit_wigner_squared(m2ab, m_res, Gamma_res):
    """
    Calculate the probability density function of the Breit-Wigner distribution squared.
    """
    gamma = Gamma_res * m_res
    bw_amplitude = 1 / ((m2ab - m_res**2)**2 + (gamma**2))
    return bw_amplitude

# Generate resonant events using the von Neumann rejection method
def generate_resonant_events_with_constraints(n, m_res, Gamma_res, m2ab_list, m2bc_list):
    resonant_m2ab = []
    resonant_m2bc = []
    
    # Maximum value of the Breit-Wigner distribution (used for normalization)
    max_bw = breit_wigner_squared(m_res**2, m_res, Gamma_res)
    
    while len(resonant_m2ab) < n:
        # Randomly pick an event from the background
        idx = np.random.randint(len(m2ab_list))
        m2ab = m2ab_list[idx]
        m2bc = m2bc_list[idx]

        # Calculate the Breit-Wigner probability density
        bw_prob_density = breit_wigner_squared(m2ab, m_res, Gamma_res)
        
        # Compare with a uniformly distributed random number scaled by the maximum BW value
        if np.random.random() < bw_prob_density / max_bw:
            resonant_m2ab.append(m2ab)
            resonant_m2bc.append(m2bc)
    print()
    return resonant_m2ab, resonant_m2bc

# Number of resonant events to generate
n_resonant = 5000

# Using previously generated background events (m2ab_data and m2bc_data) to generate resonant events
resonant_m2ab, resonant_m2bc = generate_resonant_events_with_constraints(n_resonant, m_res, Gamma_res, m2ab_data, m2bc_data)

# Combine resonant and background events
combined_m2ab = np.concatenate((resonant_m2ab, m2ab_data))
combined_m2bc = np.concatenate((resonant_m2bc, m2bc_data))

# Plot the Dalitz plot
plot_dalitz_hist2d(combined_m2ab, combined_m2bc, bins=300)

#Task 3

# Constants for the masses in GeV/c^2
m_d = 1.97  # Mass of the D_s meson
m_a = m_b = 0.498  # Mass of the Kaon
m_c = 0.135  # Mass of the Pion

# Constants for the resonances (in GeV/c^2)
m_res1 = 1.3  # Mass of the resonance in the K+ K- system
gamma_res1 = 0.015  # Decay width of the resonance in the K+ K- system
m_res2 = 1.05  # Mass of the resonance in the K- pi+ system
gamma_res2 = 0.015  # Decay width of the resonance in the K- pi+ system

def breit_wigner_complex(m2, m_res, Gamma_res):
    
    m2 = np.array(m2)  # Ensure m2 is a numpy array for element-wise operations
    gamma = Gamma_res * m_res
    return 1 / (np.complex128((m_res**2 - m2) - 1j * gamma))


def generate_interference_events(n_events, r2, theta2, m2ab, m2bc):
    m2ab = np.array(m2ab)
    m2bc = np.array(m2bc)

    # Precompute the complex amplitudes for all events
    amp_res1 = breit_wigner_complex(m2ab, m_res1, gamma_res1)
    amp_res2 = breit_wigner_complex(m2bc, m_res2, gamma_res2) * r2 * np.exp(np.complex128(1j * theta2))
    A_tot = amp_res1 + amp_res2
    
    # Calculate probability for each event
    prob = np.abs(A_tot)**2
    prob_max = np.max(prob) + 1

    events_m2ab = []
    events_m2bc = []

    while len(events_m2ab) < n_events:
        idx = np.random.randint(len(m2ab))
        
        # Random acceptance based on probability
        if np.random.uniform(0, prob_max) < prob[idx]:
            events_m2ab.append(m2ab[idx])
            events_m2bc.append(m2bc[idx])
    
    return np.array(events_m2ab), np.array(events_m2bc)


# Number of interference events to generate

n_interference = 10000 ### Should be 5000 but plot was too sparce so have increased for a nicer plot ###

# Theta values for interference (0, pi, pi/2)
theta_values = [0, np.pi, np.pi/2]

# Define the minimum and maximum values for m2ab and m2bc
m2ab_min = (m_a + m_b)**2
m2ab_max = (m_d - m_c)**2

m2bc_min = (m_b + m_c)**2
m2bc_max = (m_d - m_a)**2


def plot_dalitz(m2ab, m2bc, theta2):
    plt.figure(figsize=(8, 6), dpi=100)  # Increased figure size and dpi for higher resolution
    plt.hist2d(m2ab, m2bc, bins=300, cmap='viridis', range=[[m2ab_min, m2ab_max], [m2bc_min, m2bc_max]])  # Increased bins
    plt.colorbar()
    plt.title(f'Dalitz Plot with Interference Phase = {theta2/np.pi} $\\pi$')
    plt.xlabel('$m^2(K^+K^-)$ [GeV$^2$/c$^4$]')
    plt.ylabel('$m^2(K^-\pi^+)$ [GeV$^2$/c$^4$]')
    plt.show()

    

# Generate and plot Dalitz plots for different theta2 values
for theta2 in theta_values:
    r2 = 1
    m2ab, m2bc = generate_interference_events(n_interference, r2, theta2, m2ab_data, m2bc_data)
    plot_dalitz(np.append(m2ab, m2ab_data), np.append(m2bc, m2bc_data), theta2)

# Task 4
from scipy.optimize import minimize

m2ab_test, m2bc_test = generate_dalitz_events(50000, m_dmeson, m_kaon, m_kaon, m_pion)

# Choose pi/2
def Log_Likelihood(par, m2ab_calc, m2bc_calc):
    
    # parameters to for fitting
    r2, theta2 = par[0], par[1]
    
    
    amp_res1 = breit_wigner_complex(m2ab_calc, m_res1, gamma_res1)
    amp_res2 = breit_wigner_complex(m2bc_calc, m_res2, gamma_res2) * r2 * np.exp(np.complex128(1j * theta2))
    A_tot = amp_res1 + amp_res2
    

    # Amplitudes for normalising factor
    amp_res1 = breit_wigner_complex(m2ab_test, m_res1, gamma_res1)
    amp_res2 = breit_wigner_complex(m2bc_test, m_res2, gamma_res2) * r2 * np.exp(np.complex128(1j * theta2))
    A_tot_test = amp_res1 + amp_res2

    # Normalising Factor
    integral = np.mean(np.abs(A_tot_test)**2)
    
    norm_f = (np.abs(A_tot)**2) / integral
    
    return -np.sum(np.log(norm_f))

# Actual Values
r2_1, theta2_1 = 1, np.pi / 2

# Initial guesses
initial_guess = [0.8, 3 * np.pi / 5]

m2ab_calc, m2bc_calc = generate_interference_events(5000, r2_1, theta2_1, m2ab_data, m2bc_data)

result = minimize(Log_Likelihood, initial_guess, args=(m2ab_calc, m2bc_calc), bounds=[(0.8, 1.2), (0, 2 * np.pi)])


r2_fit, theta2_fit = result.x


print("r2:", r2_fit)
print("theta2:", theta2_fit)


def project_and_plot_1D(m2ab_data, m2bc_data, bins, fit_params):
    # Fit parameters
    r2, theta2 = fit_params
    
    # Create figure and axes for the subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Project data onto m2ab and m2bc
    counts_m2ab, edges_m2ab = np.histogram(m2ab_data, bins=bins, density=True)
    counts_m2bc, edges_m2bc = np.histogram(m2bc_data, bins=bins, density=True)
    
    # Calculate the bin centers from edges
    centers_m2ab = (edges_m2ab[:-1] + edges_m2ab[1:]) / 2
    centers_m2bc = (edges_m2bc[:-1] + edges_m2bc[1:]) / 2
    
    # Plot data projections
    axs[0].bar(centers_m2ab, counts_m2ab, width=np.diff(edges_m2ab), align='center', alpha=0.7, label='Data')
    axs[1].bar(centers_m2bc, counts_m2bc, width=np.diff(edges_m2bc), align='center', alpha=0.7, label='Data')
    
    # Generate fit result projections
    # We need to compute the probability density function for each bin center
    prob_density_m2ab = breit_wigner_squared(centers_m2ab, m_res1, gamma_res1)
    prob_density_m2bc_complex = breit_wigner_complex(centers_m2bc, m_res2, gamma_res2) * r2 * np.exp(np.complex128(1j * theta2))
    prob_density_m2bc = np.abs(prob_density_m2bc_complex)**2
    
    # Scale the probabilities by the maximum counts for visualization
    scale_factor_m2ab = counts_m2ab.max() / prob_density_m2ab.max()
    scale_factor_m2bc = counts_m2bc.max() / prob_density_m2bc.max()
    
    # Plot the fit result projections
    axs[0].plot(centers_m2ab, prob_density_m2ab * scale_factor_m2ab, color='red', label='Fit')
    axs[1].plot(centers_m2bc, prob_density_m2bc * scale_factor_m2bc, color='red', label='Fit')
    
    # Set labels and titles
    axs[0].set_xlabel('$m^2(K^+ K^-)$ (GeV$^2/c^4$)')
    axs[0].set_title('Projection on $m^2(K^+ K^-)$')
    axs[0].set_ylabel('Counts')
    axs[1].set_xlabel('$m^2(K^- \pi^+)$ (GeV$^2/c^4$)')
    axs[1].set_ylabel('Counts')
    axs[1].set_title('Projection on $m^2(K^- \pi^+)$')
    
    # Show legend
    axs[0].legend()
    axs[1].legend()
    
    # Show plot
    plt.tight_layout()
    plt.show()
    
m2ab, m2bc = generate_interference_events(n_interference, 1, np.pi/2, m2ab_data, m2bc_data)
# Perform projection and plotting
fit_params = [r2_fit, theta2_fit]  # Use the fitted values from the optimization
project_and_plot_1D(m2ab, m2bc, bins=300, fit_params=fit_params)