import shadows as sh

from credentials import API_KEY

endpoint = "http://braingeneers.gi.ucsc.edu:1337/api"

token = API_KEY
# Create a shadow object
instance = sh.DatabaseInteractor(endpoint, API_KEY)

print(instance.create_interaction_thing("test_thingy", "BioPlateScope", "test_description", {"test_key": "test_value"}))

