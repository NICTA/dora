import numpy as np
import time
from subprocess import call
import json
import hashlib


# TODO(AL) this isnt really a demo... whats going on here


def writeTreeJson(json_filename, characteristic, traits):
    # json_filename = 'jsonfiles/' + str(jobNumber) + '.json'
    args = [{'name': 'time_disturbance', 'value':  np.exp(characteristic[0])},
            {'name': 'slope', 'value': characteristic[1]},
            {'name': 'traits', 'value': [np.exp(trait) for trait in traits]}]
    jsondict = {
        'packages': ["plant", "plant.assembly"],
        'function':   "for_python_simulate",
        'args':   args,
    }
    out_file = open(json_filename, "w")
    json.dump(jsondict, out_file, indent=4)
    out_file.close()


def plant(characteristic, n_GP_slices):
    """ A binary image of a circle as a test problem for sampling
    """
    m = hashlib.md5()
    m.update(characteristic)
    jobNumber = m.hexdigest()
    json_filename = 'jsonfiles/' + str(jobNumber) + '.json'

    min_trait = -5
    max_trait = 0

    traits = np.linspace(min_trait, max_trait, n_GP_slices)
    writeTreeJson(json_filename, characteristic, traits)
    call('callr ' + json_filename, shell=True)
    with open(json_filename) as json_file:
        json_data = json.load(json_file)
    fitness = np.array(json_data['value']).astype(float)
    fitness[np.isinf(fitness)] = np.min(fitness[np.isfinite(fitness)])
    # fitness = plant_fake(characteristic, n_GP_slices)

    return fitness

def plant_fake(X, n_GP_slices):
    """ A binary image of a circle as a test problem for sampling
    """
    fitness_range = [0, 1]
    centre1 = np.array([1.5, 1.4, 0.3])
    centre2 = np.array([2.50, 2.0, 0.7])
    l1 = 0.2
    l2 = 0.1
    Z = np.arange(fitness_range[0], fitness_range[1],
                  (fitness_range[1] - fitness_range[0])/n_GP_slices)
    dist1 = [np.sqrt((centre1[0]-X[0])**2+(centre1[1]-X[1])**2
                     + (centre1[2]-z)**2) for z in Z]
    dist2 = [np.sqrt((centre2[0]-X[0])**2+(centre2[1]-X[1])**2
                     + (centre2[2]-z)**2) for z in Z]
    fitness = np.asarray([np.exp(-dist1[i]/l1) + np.exp(-dist2[i]/l2)
                          for i in range(len(dist1))]) #- 0.9 +0.785
    return fitness

