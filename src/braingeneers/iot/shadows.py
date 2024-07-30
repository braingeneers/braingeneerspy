import requests
import configparser
import os
import io
from typing import Union


class DatabaseInteractor:
    """
    This class provides methods for interacting with the Strapi Shadows database.

    See documentation at: ...

    Assumes the following:
        - ~/.aws/credentials file exists and contains a section [strapi] with the following keys:
            - endpoint: the URL of the Strapi server
            - api_key: the API key for the Strapi server

    class objects:
        - __API_object: base class for all objects in the database
        - __Thing: class for interaction thing objects
        - __Experiment: class for experiment objects
        - __Plate: class for plate objects
        - __Well: class for well objects
        - __Sample: class for sample objects

        
    Public functions:
        - empty_trash: deletes all objects with attribute marked_for_deletion set to True
        - create_interaction_thing: creates a new interaction thing object in the database
        - create_plate: creates a new plate object in the database
        - create_experiment: creates a new experiment object in the database
        - start_image_capture: updates the shadow of the interaction thing with the uuid of the image run
        - list_objects: returns a list of objects of a given type
        - list_objects_with_name_and_id: returns a list of dictionaries with the name and id of each object
        - list_experiments: returns a list of experiment names
        - list_BioPlateScopes: returns a list of BioPlateScope names and ids
        - list_devices_by_type: returns a list of device names and ids of a given type
        - get_device_state: returns the shadow of a device given its id
        - get_device_state_by_name: returns the shadow of a device given its name
        - get_device: returns a device object given its id or name
        - get_plate: returns a plate object given its id
        - get_experiment: returns an experiment object given its id
        - get_sample: returns a sample object given its id
        - get_well: returns a well object given its id
    """
    def __init__(self, credentials: Union[str, io.IOBase] = None, overwrite_endpoint=None, overwrite_api_key=None, jwt_service_token=None) -> None:

        if credentials is None:
            credentials = os.path.expanduser('~/.aws/credentials')  # default credentials location

        if isinstance(credentials, str):
            with open(credentials, 'r') as f:
                self._credentials = f.read()
        else:
            assert hasattr(credentials, 'read'), 'credentials parameter must be a filename string or file-like object.'
            self._credentials = credentials.read()

        config = configparser.ConfigParser()
        config.read_file(io.StringIO(self._credentials))
        assert 'strapi' in config, 'Your AWS credentials file is missing a section [strapi], ' \
                                    'you may have the wrong version of the credentials file.'
        assert 'endpoint' in config['strapi'], 'Your AWS credentials file is malformed, ' \
                                                'endpoint was not found under the [strapi] section.'
        assert 'api_key' in config['strapi'], 'Your AWS credentials file is malformed, ' \
                                                'api_key was not found under the [strapi] section.'

        # Note that the "token" is a basic auth construct originally implemented with Strapi before full JWT
        # authentication was available. It's deprecated, but still in use, if someone wants to reconfigure
        # the services to remove it that would be good, but until then it's a superfluous detail.
        # The JWT service token is the updated way to authenticate with all web services including Strapi
        self.endpoint = config['strapi']['endpoint']
        self.token = config['strapi']['api_key']
        if overwrite_endpoint:
            self.endpoint = overwrite_endpoint
        if overwrite_api_key:
            self.token = overwrite_api_key
        self.jwt_service_token = jwt_service_token

    class __API_object:
        """
        This class is used to represent objects in the database as python objects
        """
        def __init__(self, endpoint, api_token, api_object_id, jwt_service_token):
            self.endpoint = endpoint
            self.token = api_token
            self.id = None
            self.attributes = {}
            self.api_object_id = api_object_id
            self.jwt_service_token = jwt_service_token

        def __str__(self):
            var_list = filter(lambda x: x not in ["endpoint", "token", "api_object_id"], vars(self))
            return str({var: getattr(self, var) for var in var_list})

        def to_json(self):
            var_list = filter(lambda x: x not in ["endpoint", "token", "api_object_id"], vars(self))
            return {var: getattr(self, var) for var in var_list}

        def parse_API_response(self, response_data):
            """
            parses the response from the API and updates the python object
            """
            self.id = response_data['id']
            self.attributes = response_data['attributes']
            for key in self.attributes:
                if type(self.attributes[key]) is dict and "data" in self.attributes[key]:
                    if self.attributes[key]["data"] is not None and len(self.attributes[key]["data"]) != 0:
                        item_list = []
                        if type(self.attributes[key]["data"]) is list:
                            for item in self.attributes[key]["data"]:
                                if "id" in item:
                                    item_list.append(item["id"])
                        else:
                            item_list.append(self.attributes[key]["data"]["id"])

                        self.attributes[key] = item_list
                    else:
                        self.attributes[key] = []

        def spawn(self):
            """
            creates a new object in the database
            """
            url = self.endpoint + "/"+self.api_object_id+"?filters[name][$eq]=" + self.attributes["name"] + "&populate=%2A"
            headers = {"Authorization": "Bearer " + self.jwt_service_token['access_token']}
            response = requests.get(url, headers=headers)
            if len(response.json()['data']) == 0:
                api_url = self.endpoint+"/"+self.api_object_id+"?populate=%2A"
                data = {"data": self.attributes}
                response = requests.post(api_url, json=data, headers=headers)
                if response.status_code == 200:
                    self.parse_API_response(response.json()['data'])
            else:
                print(self.api_object_id + " object already exists")
                try:
                    self.parse_API_response(response.json()['data'][0])
                except KeyError:
                    print("some values are missing")

        def push(self):
            """
            updates the database with the current state of the object
            """
            url = self.endpoint + "/"+self.api_object_id+"/" + str(self.id) + "?populate=%2A"
            headers = {"Authorization": "Bearer " + self.jwt_service_token['access_token']}
            data = {"data": self.attributes}
            response = requests.put(url, headers=headers, json=data)
            self.parse_API_response(response.json()['data'])

        def pull(self):
            """
            updates object with the latest data from the database
            """
            url = self.endpoint + "/"+self.api_object_id+"/" + str(self.id) + "?populate=%2A"
            headers = {"Authorization": "Bearer " + self.jwt_service_token['access_token']}
            response = requests.get(url, headers=headers)
            if len(response.json()['data']) == 0:
                raise Exception("Object not found")

            self.parse_API_response(response.json()['data'])

        def get_by_name(self, name):
            """
            gets the object from the database by name
            """
            url = self.endpoint + "/"+self.api_object_id+"?filters[name][$eq]=" + name + "&populate=%2A"
            headers = {"Authorization": "Bearer " + self.jwt_service_token['access_token']}
            response = requests.get(url, headers=headers)
            if len(response.json()['data']) == 0:
                raise Exception("no " + self.api_object_id + " object with name " + name)
            else:
                self.parse_API_response(response.json()['data'][0])
          
        def move_to_trash(self):
            """
            marks the object for deletion
            """
            url = self.endpoint + "/"+self.api_object_id+"/" + str(self.id) + "?populate=%2A"
            headers = {"Authorization": "Bearer " + self.jwt_service_token['access_token']}
            response = requests.get(url, headers=headers)
            if len(response.json()['data']) == 0:
                raise Exception("Object not found")
            else:
                self.parse_API_response(response.json()['data'])
                self.attributes["marked_for_deletion"] = True
                self.push()

        def recover_from_trash(self):
            """
            unmarks the object for deletion
            """
            url = self.endpoint + "/"+self.api_object_id+"/" + str(self.id) + "?populate=%2A"
            headers = {"Authorization": "Bearer " + self.jwt_service_token['access_token']}
            response = requests.get(url, headers=headers)
            if len(response.json()['data']) == 0:
                raise Exception("Object not found")
            else:
                self.id = response.json()['data']['id']
                self.attributes = response.json()['data']['attributes']
                self.attributes["marked_for_deletion"] = False
                self.push()

    class __Thing(__API_object):
        def __init__(self, endpoint, api_token, jwt_service_token):
            super().__init__(endpoint, api_token, "interaction-things", jwt_service_token)

        def add_to_shadow(self, json):
            if self.attributes["shadow"] is None:
                self.attributes["shadow"] = json
            else:
                for key, value in json.items():
                    self.attributes["shadow"][key] = value

            self.push()

        def add_uuid_to_shadow(self, uuid):
            if self.attributes["shadow"] is None:
                self.attributes["shadow"] = {}
            self.attributes["shadow"]["uuid"] = uuid
            self.push()            

        def set_current_plate(self, plate):
            """
            updates the current plate of the thing and adds the plate to the list of all plates historically associated with the thing.
            The plates list relation also updates the thing relation on the plate object itself. 
            """

            if self.attributes["plates"] is None:
                self.attributes["plates"] = []
            self.attributes["plates"].append(plate.id)
            self.attributes["current_plate"] = plate.id

            plate.add_thing(self)
            self.push()

        def set_current_experiment(self, experiment):
            """

            updates the current experiment of the thing
            """
            self.attributes["current_experiment"] = experiment.id
            self.push()

    class __Experiment(__API_object):
        def __init__(self, endpoint, api_token, jwt_service_token):
            super().__init__(endpoint, api_token, "experiments", jwt_service_token=jwt_service_token)

        def add_plate(self, plate):
            # Bidirectional relations have an owner and a related object, plate owns this relation
            plate.add_experiment(self)
            self.pull()

    class __Plate(__API_object):
        def __init__(self, endpoint, api_token, jwt_service_token):
            super().__init__(endpoint, api_token, "plates", jwt_service_token)

        def add_thing(self, thing):
            """
            add_thing_to_plate

            adds the thing to the list of things associated with the plate. also updates the plate relation on the thing object itself.
            does not effect current_plate value of thing
            """
            if self.attributes["interaction_things"] is None:
                self.attributes["interaction_things"] = []
            self.attributes["interaction_things"].append(thing.id)
            self.push()

        def add_uuid_to_image_params(self, uuid):
            if self.attributes["image_parameters"] is None:
                self.attributes["image_parameters"] = {}

            # if attributes["image_parameters] has attribute uuid" and it is not none
            if "uuids" not in self.attributes["image_parameters"] or self.attributes["image_parameters"]["uuids"] is None:
                self.attributes["image_parameters"]["uuids"] = {}
            for key, value in uuid.items():
                self.attributes["image_parameters"]["uuids"][key] = value

            self.push()

        def add_entry_to_ephys_params(self, uuid, channels, timestamp, data_length):
            if self.attributes["ephys_parameters"] is None:
                self.attributes["ephys_parameters"] = {}
              
            # structure is {"uuids" : {uuid: {"channels": {channel : {"timestamps" : {timestamp: "length : [data_length]}}}

            if "uuids" not in self.attributes["ephys_parameters"] or self.attributes["ephys_parameters"]["uuids"] is None:
                self.attributes["ephys_parameters"]["uuids"] = {}
            if uuid not in self.attributes["ephys_parameters"]["uuids"]:
                self.attributes["ephys_parameters"]["uuids"][uuid] = {}
            if "channels" not in self.attributes["ephys_parameters"]["uuids"][uuid] or self.attributes["ephys_parameters"]["uuids"][uuid]["channels"] is None:
                self.attributes["ephys_parameters"]["uuids"][uuid]["channels"] = {}
            if channels not in self.attributes["ephys_parameters"]["uuids"][uuid]["channels"]:
                self.attributes["ephys_parameters"]["uuids"][uuid]["channels"][channels] = {}
            if "timestamps" not in self.attributes["ephys_parameters"]["uuids"][uuid]["channels"][channels] or self.attributes["ephys_parameters"]["uuids"][uuid]["channels"][channels]["timestamps"] is None:
                self.attributes["ephys_parameters"]["uuids"][uuid]["channels"][channels]["timestamps"] = {}
            if timestamp not in self.attributes["ephys_parameters"]["uuids"][uuid]["channels"][channels]["timestamps"]:
                self.attributes["ephys_parameters"]["uuids"][uuid]["channels"][channels]["timestamps"][timestamp] = {}
            if "length" not in self.attributes["ephys_parameters"]["uuids"][uuid]["channels"][channels]["timestamps"][timestamp] or self.attributes["ephys_parameters"]["uuids"][uuid]["channels"][channels]["timestamps"][timestamp]["length"] is None:
                self.attributes["ephys_parameters"]["uuids"][uuid]["channels"][channels]["timestamps"][timestamp]["length"] = []
            if data_length not in self.attributes["ephys_parameters"]["uuids"][uuid]["channels"][channels]["timestamps"][timestamp]["length"]:
                self.attributes["ephys_parameters"]["uuids"][uuid]["channels"][channels]["timestamps"][timestamp]["length"].append(data_length)

            self.push() 

        def add_experiment(self, experiment):
            if self.attributes["experiments"] is None:
                self.attributes["experiments"] = []
            self.attributes["experiments"].append(experiment.id)
            self.push()

    class __Well(__API_object):
        def __init__(self, endpoint, api_token, jwt_service_token):
            super().__init__(endpoint, api_token, "wells", jwt_service_token)
        
    class __Sample(__API_object):
        def __init__(self, endpoint, api_token, jwt_service_token):
            super().__init__(endpoint, api_token, "samples", jwt_service_token)

    def empty_trash(self):
        """
        delete all objects with attribute marked_for_deletion set to True
        """
        object_list = ["interaction-things", "experiments", "plates", "wells", "samples"]
        for object in object_list:
          url = self.endpoint + "/"+object+"?filters[marked_for_deletion][$eq]=true&populate=%2A"
          headers = {"Authorization": "Bearer " + self.jwt_service_token['access_token']}
          response = requests.get(url, headers=headers)
          for item in response.json()['data']:
              url = self.endpoint + "/"+object+"/" + str(item['id'])
              response = requests.delete(url, headers=headers)
              print("deleted object of type: " + object + " with id " + str(item['id']))
        
    def create_interaction_thing(self, type, name):
        thing = self.__Thing(self.endpoint, self.token, self.jwt_service_token)
        thing.attributes["name"] = name
        thing.attributes["type"] = type
        thing.spawn()
        return thing

    def create_plate(self, name, rows, columns, image_params={}):
        plate = self.__Plate(self.endpoint, self.token, self.jwt_service_token)
        plate.attributes["name"] = name
        plate.attributes["rows"] = rows
        plate.attributes["columns"] = columns
        plate.attributes["image_parameters"] = image_params
        plate.spawn()

        if len(plate.attributes["wells"]) == 0 or plate.attributes["wells"] is None:
            for i in range(1, rows+1):
                for j in range(1, columns+1):
                    well = self.__Well(self.endpoint, self.token, self.jwt_service_token)
                    well.attributes["name"] = plate.attributes["name"]+"_well_"+str(i)+str(j)
                    well.attributes["position_index"] = str(i) + str(j)
                    well.attributes["plate"] = plate.id
                    well.spawn()

        plate.pull()
        return plate
    
 
    def create_experiment(self, name, description):
        experiment = self.__Experiment(self.endpoint, self.token, self.jwt_service_token)
        experiment.attributes["name"] = name
        experiment.attributes["description"] = description
        experiment.spawn()
        return experiment
        

    def start_image_capture(self, thing, uuid):
        thing.add_uuid_to_shadow(uuid)
        group_id = thing.attributes["shadow"]["group-id"]
        value = { uuid : group_id }
        if thing.attributes["current_plate"]:
            plate = self.__Plate(self.endpoint, self.token, self.jwt_service_token)
            plate.id = thing.attributes["current_plate"][0]
            plate.pull()
            plate.add_uuid_to_image_params(value)
        else: 
            raise Exception("no plate associated with thing")
 
    def list_objects(self, api_object_id, filter="?", hide_deleted=True):
        """
        when you need a list of the objects in the database
        useful for populating dropdown lists in plotly dash
        """
        if hide_deleted:
            filter += "&filters[marked_for_deletion][$eq]=false"
        url = self.endpoint + "/"+  api_object_id + filter +"&populate=%2A"
        headers = {"Authorization": "Bearer " + self.jwt_service_token['access_token']}
        response = requests.get(url, headers=headers)
        return response.json()['data']

    def list_objects_with_name_and_id(self, api_object_id, filter="?", hide_deleted=True):
        """
        when you need a list of the objects in the database

        returns a list of dictionaries with the name and id of each object

        """
        response = self.list_objects(api_object_id, filter, hide_deleted)
        return [{"label": x["attributes"]["name"], "value": x["id"]} for x in response]

    def list_experiments(self, hide_deleted = True):
        response = self.list_objects("experiments", "?", hide_deleted)
        output = []
        for i in response:
            output.append(i["attributes"]["name"])
        return output

    def list_BioPlateScopes(self, hide_deleted=True):
        return self.list_objects_with_name_and_id("interaction-things", "?filters[type][$eq]=BioPlateScope", hide_deleted)

    def list_devices_by_type(self, thingTypeName, hide_deleted = True):
        return self.list_objects_with_name_and_id("interaction-things", "?filters[type][$eq]="+thingTypeName, hide_deleted)

    def get_device_state(self, thing_id):
        thing = self.__Thing(self.endpoint, self.token, self.jwt_service_token)
        thing.id = thing_id
        thing.pull()
        return thing.attributes["shadow"]

    def get_device_state_by_name(self, thing_name):
        thing = self.__Thing(self.endpoint, self.token, self.jwt_service_token)
        thing.get_by_name(thing_name)
        return thing.attributes["shadow"]



    """
    Getters for objects from their id numbers
    """

    def get_device(self, thing_id=None, name=None):
        if thing_id is None and name is None:
            raise Exception("must provide either thing_id or name")
        if name:
            thing = self.__Thing(self.endpoint, self.token, self.jwt_service_token)
            thing.get_by_name(name)
            return thing
        else:
            thing = self.__Thing(self.endpoint, self.token, self.jwt_service_token)
            thing.id = thing_id
            thing.pull()
            return thing
    
    def get_plate(self, plate_id):
        plate = self.__Plate(self.endpoint, self.token, self.jwt_service_token)
        plate.id = plate_id
        plate.pull()
        return plate

    def get_experiment(self, experiment_id):
        experiment = self.__Experiment(self.endpoint, self.token, self.jwt_service_token)
        experiment.id = experiment_id
        experiment.pull()
        return experiment

    def get_sample(self, sample_id):
        sample = self.__Sample(self.endpoint, self.token, self.jwt_service_token)
        sample.id = sample_id
        sample.pull()
        return sample

    def get_well(self, well_id):
        well = self.__Well(self.endpoint, self.token, self.jwt_service_token)
        well.id = well_id
        well.pull()
        return well
