import requests
import time

# from credentials import API_KEY
# self.endpoint = "http://braingeneers.gi.ucsc.edu:1337/api"
# API_KEY = "boof"
# self.token = API_KEY

class DatabaseInteractor:
    
    def __init__(self, endpoint, api_token) -> None:
        self.endpoint = endpoint
        self.token = api_token

    def create_interaction_thing(self, name, interaction_type, description="", shadow={}):
        api_url = self.endpoint + "/interaction-things/"
        headers = {"Authorization": "Bearer " + self.token}
        info = {
            "data": {
                "name": name,
                "description": description,
                "type": interaction_type,
                "shadow": shadow
            }
        }

        response = requests.post(self, api_url, headers=headers, json=info)
        # response = requests.post(api_url, json=info, headers={
        #                         'Authorization': 'bearer ' + self.token})
        return response.json()


    def update_experiment_on_interaction_thing(self, interaction_thing_id, experiment_id):
        url = self.endpoint + "/interaction-things/" + str(interaction_thing_id)
        headers = {"Authorization": "Bearer " + self.token}
        data = {
            "experiment_id": experiment_id
        }
        response = requests.put(url, headers=headers, json=data)
        return response


    def update_plate_on_interaction_thing(self, interaction_thing_id, plate_id):
        url = self.endpoint + "/interaction-things/" + str(interaction_thing_id)
        headers = {"Authorization": "Bearer " + self.token}
        data = {
            "plate_id": plate_id
        }
        response = requests.put(url, headers=headers, json=data)
        return response

    def get_interaction_thing_id_from_name(self, name):
        url = self.endpoint + "/interaction-things?filters[name][$eq]=" + name
        headers = {"Authorization": "Bearer " + self.token}
        response = requests.get(url, headers=headers)
        print(response.json())
        return response.json()['data'][0]['id']

    def update_shadow_without_overwrite(self, interaction_thing_id, shadow):
        url = self.endpoint + "/interaction-things/" + str(interaction_thing_id)
        headers = {"Authorization": "Bearer " + self.token}
        data = {
            "shadow": shadow
        }
        response = requests.put(url, headers=headers, json=data)
        return response

    def list_all_interaction_things(self):
        url = self.endpoint + "/interaction-things"
        headers = {"Authorization": "Bearer " + self.token}
        response = requests.get(url, headers=headers)
        return response.json()
