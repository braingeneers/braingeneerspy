""" A simple protocol for connecting devices to IoT and schedule experiments"""

# import stuff
from braingeneers.iot import messaging
import uuid
import schedule
import time
import warnings
import builtins
import inspect
import logging
import traceback
from dateutil import tz


def start_iot(device_name, device_type, experiment, commands=[]): #, #allowed_commands={"pump":["func_1","func_1"], "image":["func_1","func_1"]}):
    """Create a device and have it start listening for commands. This is intended for simple use cases"""
    
    # Run Helper Funcitons
    def _updateIot(device_name, mb ):
        """Updates the IoT device's publicly visible state according to the local schedule/status"""
        # Update schedule
        try:                                                                     # Use try/catch so that error like no internet don't stop experiment
            jobs, jobs_str = [],[]
            for i,job in enumerate(schedule.get_jobs()):                         # debugging: print(i) #print(job.__str__()) #print(job.__repr__())
                jobs.append( job.__str__() )                                     # append python command that wass used to create shceduled job
                if job.cancel_after:                                             # if job must be cancelled at a certain time, add cancel time
                    jobs[i]+= job.cancel_after.astimezone(tz.gettz('US/Pacific')).strftime("UNTIL-%Y-%m-%d %H:%M:%S") 

                job_str = f"Job {i}: "+job.__repr__()
                if job.last_run:                                                                                # if job was previously run
                    last_time = job.last_run.astimezone(tz.gettz('US/Pacific')).strftime("%Y-%m-%d %H:%M:%S")   # get time of last scheduled event
                    job_str = job_str.split("last run: ")[0]+"last run: "+last_time+", next run: "
                next_time = job.next_run.astimezone(tz.gettz('US/Pacific')).strftime("%Y-%m-%d %H:%M:%S)")      # get time of next scheduled event     
                jobs_str.append( job_str.split("next run: ")[0]+"next run: "+next_time )# append

            # Update log
            with open("iot.log") as file:                  # Get history of iot commands and scheduel from the file iot.log
                lines = file.readlines()
                log = [line.rstrip() for line in lines]
                if len(log)>35:                            # If iot log is more than 25 lines, get just the last 25 lines
                    log = log[-35:]    
            mb.update_device_state( device_name, {"status":iot_status,"jobs":jobs,"schedule":jobs_str,"history":log} )
            
        except:
            logger.error("\n"+traceback.format_exc())                             # Write error to log
    
    
    # Create function for when IoT command is received
    def respondToCommand(topic: str, message: dict):                               # build function that runs when device receives command
        global last_mqtts
        try:                                                                       #DEBUG: print("command Received") #print(message)
            if len(commands)>0:                                                    # if user inputed list of allowed commands 
                if not "global iot_status; iot_status=" in message["command"] and not any(x in message["command"] for x in commands ):            # check if sent command contains an allowed command, if not throw error
                    raise Exception(f"User message-- {message['command']}-- doesn't contain required commands -- {commands}")         
            if  message["id"] in last_mqtts :
                raise Exception(f"User message-- {message['command']}-- received multiple times and was stopped from running again")
            last_mqtts = last_mqtts[1:] + [message["id"]]
            logger.debug(f"Run Command: {message['command']}")                     # log to history that the sent command was run
            exec(message["command"])                                               # run python command that was sent
        except Exception as e:
            logger.error("\n"+traceback.format_exc())
        _updateIot(device_name, mb )                                               # update iot state, in case schedule/status changed

    
    # Initialize environment
    from braingeneers.iot import messaging; import uuid; import schedule; import time; import warnings; import logging; import traceback #requried packages
    global iot_status                                                              # states if iot is running. Other iot functions can change it asynchronously
    global last_mqtts
    warnings.filterwarnings(action='once')                                         # stops same warning from appearing more than once
    mb = messaging.MessageBroker(str(uuid.uuid4))                                  # spin up iot
    last_mqtts = [""]*200 
    
    # Set up Logging
    open("iot.log", "w").close()
    logging.basicConfig( level=logging.WARNING ) 
    logger = logging.getLogger('schedule')
    logger.setLevel(level=logging.DEBUG)
    f_handler = logging.FileHandler('iot.log')
    f_handler.setFormatter( logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S %Z") ) # %(levelname)s
    logger.addHandler(f_handler)
    global print
    print = logger.debug

    # Start IoT Device
    if device_name not in mb.list_devices_by_type(thing_type_name= device_type):    # check if device already exists
        mb.create_device( device_name= device_name, device_type= device_type)      # if not, create it
    else:                                                                          # otherwise, check device is ok and isn't still running
        assert "status" in mb.get_device_state(device_name), f"{device_name} has corrupted data! Talk to data team."
        assert mb.get_device_state(device_name)["status"]=="shutdown", f"{device_name} already exists and isn't shutdown. Please shutdown with 'iot.shutdown({device_name})'"
    mb.update_device_state( device_name, {"experiment":experiment,"status":"run","jobs":[],"schedule":[],"history":[]} ) # initialize iot state
    mb.subscribe_message( f"devices/+/{device_name}", respondToCommand )           # start listening for new commands
    #mb.subscribe_message(f"devices/{device_type}/{device_name}",respondToCommand)  
    
    # Perpetually listen for IoT commands
    iot_status = "run"                                                             # keep python running so that listener can do it's job
    while not iot_status=="shutdown":                                              # when it's time to stop, iot makes iot_status='shutdown'{}
        if iot_status=="run":                                                      # if the device is in run mode,
            is_pending = sum([job.should_run for job in schedule.jobs])            # Get the number of pending jobs that should be run now
            if is_pending:                                                         # if there are any pending jobs to run
                schedule.run_pending()                                             # run the pending jobs
                _updateIot(device_name, mb )                                       # schedule info has changed, so update IoT state
        time.sleep(.1)                                                             # wait a little to save cpu usage
    mb.shutdown()                                                                  # shutdown iot at the end.
    
    
def ready_iot():
    """Save source code for start_iot function to a place where it can be executed by the user"""
    builtins.ready_iot = inspect.getsource(start_iot)

