import numpy as np
import glob
import subprocess
import matplotlib.pyplot as plt


def plot_for_model(model):
    filenames = glob.glob("../results/decay_*channel4_{}.log".format(model))
    masses  = []
    results = []
    
    for filename in filenames:
        try:
            line   = subprocess.Popen("grep 'CL norm' {}".format(filename), shell=True, stdout=subprocess.PIPE).stdout.read()
            print(line)
            mass   = float(filename.split("/")[-1].split("mass")[-1].split("GeV")[0])
            result = float(line.split()[-1])
            masses.append(mass)
            results.append(result)
        #print(mass, result)
        except:
            continue

    masses = np.array(masses)
    results = np.array(results)
    
    sorted_indices = np.argsort(masses)
    plt.loglog(masses[sorted_indices], results[sorted_indices], label=r'Virgo Cluster - $\tau^+\tau^-$ [extended] - {}'.format(model))
        
def plot_dSph_results():
    combined_filename = "/home/tyapici/Projects/HAWC_projects/HAWC_dSph_DarkMatter/results/txt/decay/Combined_wo_TriII_tautau.dat"
    masses, results = np.loadtxt(combined_filename, unpack=True)
    sorted_indices = np.argsort(masses)
    plt.loglog(masses[sorted_indices], results[sorted_indices], label=r'dSph w/o TriII $\tau^+\tau^-$')
    
    combined_filename = "/home/tyapici/Projects/HAWC_projects/HAWC_dSph_DarkMatter/results/txt/decay/Combined_w_TriII_tautau.dat"
    masses, results = np.loadtxt(combined_filename, unpack=True)
    sorted_indices = np.argsort(masses)
    plt.loglog(masses[sorted_indices], results[sorted_indices], label=r'dSph w/ TriII $\tau^+\tau^-$')

    
plot_for_model("GAO")
plot_for_model("B01")
plot_dSph_results()
plt.xlim(1e3, 1e5)
#plt.ylim(1e-25, 1e-20)
plt.xlabel(r"$M_\chi$ [GeV]")
plt.ylabel(r"<$\sigma v$>")
plt.legend()
plt.savefig("../results/95CL_limits/annihilation_results.pdf")
plt.show()

