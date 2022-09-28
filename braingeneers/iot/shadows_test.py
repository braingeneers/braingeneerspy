import shadows as sh

from credentials import API_KEY

endpoint = "http://braingeneers.gi.ucsc.edu:1337/api"

token = API_KEY
# Create a shadow object
instance = sh.DatabaseInteractor(endpoint, API_KEY)

thing = instance.create_interaction_thing("BioPlateScope", "full_test_thing4")

print(thing)

thing.add_to_shadow("barg","sparkleeness")

print(thing)

# print(thing)

thing = instance.update_thing_on_database(thing)

print("updated thing: ", thing)

experiment = instance.create_experiment("test_experiment_obj_4", "test_description")
# print(type(experiment))

thing = instance.add_experiment_to_thing(thing, experiment)
print("added experiment", thing)
# thing.add_to_shadow("fart","sparkle")

# print(thing.to_json())

# thing.add_to_shadow("barg","sparkleeness")
# thing.add_to_shadow("barg",{"nuts":"sparkle"})
# print(thing.to_json())
# print(instance.create_interaction_thing("test_thingy", "BioPlateScope", "test_description", {"test_key": "test_value"}))

# thing = instance.get_thing_from_database("Forky")
# print(thing.to_json())
# thing.add_to_shadow("fartt","sparklettt")
# print(thing.to_json())
# instance.update_thing_on_database(thing)

# plate = instance.create_plate("test_plate_obj", 2, 4)
# print(plate.to_json())
# instance.sync_plate(plate)
# plate = instance.get_plate(10)
# print(plate)
# experiment = instance.create_experiment("test_experiment_obj_3", "test_description")
# experiment = instance.get_experiment(9)
# print(experiment)
# experiment.plates.append(10)
# experiment.plates.append(6)

# instance.sync_experiment(experiment)
# experiment = instance.get_experiment(9)
# print(experiment)
'''
How do we want this to flow

Create interaction thing
    - only happens once, when a new device is added to the system
    - contains important imaging data

Create Experiment
    - happens when a new experiment is started
    - Creates a plate or assigns existing plate to experiment
    - assigns current experiment to interaction thing?

Create Plate
    - Plate spawns wells, wells don't exist outside of plates
    - Plates have interaction things
    - Imaging metadata should be transferred to the plate from the interaction thing

Starting an image run
    - Image uuid must be passed to plate object
    - should not be able to start image run if interaction thing has no plate


'''

