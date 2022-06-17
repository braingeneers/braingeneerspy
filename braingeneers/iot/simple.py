""" A simple protocol for connecting devices to IoT and schedule experiments"""



# import stuff
from braingeneers.iot import messaging
import uuid
import schedule
import time
import warnings




def start(device_name, device_type, experiment):
    """Create a device and have it start listening for commands. This is intended for simple use cases"""
    global iot_status                                                              # states if iot is running. Other iot functions can change it asynchronously
    warnings.filterwarnings(action='once')                                         # stops same warning from appearing more than once
    mb = messaging.MessageBroker(str(uuid.uuid4))                                  # spin up iot
    
    if device_name not in mb.list_devices_by_type( thingTypeName= device_type):    # check if device already exists
        mb.create_device( device_name= device_name, device_type= device_type)      # if not, create it
    else:                                                                          # otherwise, check device is ok and isn't still running
        assert "status" in mb.get_device_state(device_name), f"{device_name} has corrupted data! Talk to data team."
        assert mb.get_device_state(device_name)["status"]=="shutdown", f"{device_name} already exists and isn't shutdown. Please shutdown with 'iot.shutdown({device_name})'"
    mb.update_device_state( device_name, {"experiment":experiment,"status":"run","schedule":[]} )    # reset state for new run
    
    def respondToCommand(topic: str, message: dict):                               # build function that runs when device receives command
        exec(message["command"])                                                   # run python command that was sent
        schedule_str= [f"Job {i}: "+x.__repr__() for i,x in enumerate(schedule.get_jobs())]  # turn schedule into list of strings
        mb.update_device_state( device_name, {"schedule":schedule_str, "status":iot_status} )    # in case schedule or status changed, update state's schedule
    #mb.subscribe_message( f"devices/{device_type}/{device_name}", respondToCommand )   # start listening for new commands
    mb.subscribe_message( f"devices/+/{device_name}", respondToCommand )   # start listening for new commands
    
    iot_status = "run"                                                             # keep python running so that listener can do it's job
    while not iot_status=="shutdown":                                              # when it's time to stop, iot makes iot_status='shutdown'{}
        if iot_status=="run":                                                      # if the device is in run mode,
            schedule.run_pending()                                                 # run any scheduled commands if it's their time
        time.sleep(.1)                                                             # wait a little to save cpu usage
    mb.shutdown()                                                                  # shutdown iot at the end.


def send( device_name, command ):
    """Send a python script as a string which is then implemented by an IoT device. This is intended for simple use cases"""
    warnings.filterwarnings("ignore")
    mb = messaging.MessageBroker(str(uuid.uuid4))                             # spin up iot
    mb.publish_message( topic=f"devices/dummy/{device_name}", message={"command": command } )    # send command to listening device
    mb.shutdown()                                                             # shutdown iot


def get_schedule( device_name ):
    """Get a list of scheduled commands from a device's shadow. This is intended for simple use cases"""
    warnings.filterwarnings("ignore")
    mb = messaging.MessageBroker(str(uuid.uuid4))                             # spin up iot
    my_schedule = mb.get_device_state( device_name )["schedule"]              # get schedule for device
    mb.shutdown()                                                             # shutdown iot
    return my_schedule                                                        # return schedule to user


def get_status( device_name ):
    """Get a list of scheduled commands from a device's shadow. This is intended for simple use cases"""
    warnings.filterwarnings("ignore")
    mb = messaging.MessageBroker(str(uuid.uuid4))                        # spin up iot
    status = mb.get_device_state( device_name )["status"]                # get schedule for device
    mb.shutdown()                                                        # shutdown iot
    return status                                                        # return schedule to user


def shutdown( device_name ):
    """Stops iot listener on device by changing flag on shadow. This is intended for simple use cases"""
    send(device_name, "global iot_status; iot_status='shutdown'")
    # the following lines of code may or may not be good to add, I haven't decided yet
    #warnings.filterwarnings("ignore") #mb = messaging.MessageBroker(str(uuid.uuid4)) #mb.update_device_state( device_name, {"status":"shutdown"} ) #mb.shutdown()
    
    
def pause( device_name ):
    """Pauses iot listener on device by changing flag on shadow. This is intended for simple use cases"""
    send(device_name, "global iot_status; iot_status='pause'")

    
def run( device_name ):
    """Resumes running of iot listener on device by changing flag on shadow. This is intended for simple use cases"""
    send(device_name, "global iot_status; iot_status='run'")




