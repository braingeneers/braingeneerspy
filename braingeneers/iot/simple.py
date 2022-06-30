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

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from dateutil import tz


def _updateIot(device_name, mb ):
    """Updates the IoT device's publicly visible state according to the local schedule/status"""
    # Update schedule
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



def start_iot(device_name, device_type, experiment, commands=[]): #, #allowed_commands={"pump":["func_1","func_1"], "image":["func_1","func_1"]}):
    """Create a device and have it start listening for commands. This is intended for simple use cases"""
    # Initialize environment
    from braingeneers.iot import messaging; import uuid; import schedule; import time; import warnings; import logging; import traceback #requried packages
    global iot_status                                                              # states if iot is running. Other iot functions can change it asynchronously
    warnings.filterwarnings(action='once')                                         # stops same warning from appearing more than once
    mb = messaging.MessageBroker(str(uuid.uuid4))                                  # spin up iot
    
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
    if device_name not in mb.list_devices_by_type( thingTypeName= device_type):    # check if device already exists
        mb.create_device( device_name= device_name, device_type= device_type)      # if not, create it
    else:                                                                          # otherwise, check device is ok and isn't still running
        assert "status" in mb.get_device_state(device_name), f"{device_name} has corrupted data! Talk to data team."
        assert mb.get_device_state(device_name)["status"]=="shutdown", f"{device_name} already exists and isn't shutdown. Please shutdown with 'iot.shutdown({device_name})'"
    mb.update_device_state( device_name, {"experiment":experiment,"status":"run","jobs":[],"schedule":[],"history":[]} ) # initialize iot state
    
    # Create function for when IoT command is received
    def respondToCommand(topic: str, message: dict):                               # build function that runs when device receives command
        try:
            if len(commands)>0:                                                    # if user inputed list of allowed commands 
                if not "global iot_status; iot_status=" in message["command"] and \ 
                   not any(x in message["command"] for x in commands ):            # check if sent command contains an allowed command, if not throw error
                    raise Exception(f"User message-- {message['command']}-- doesn't contain required commands -- {commands}")         
            logger.debug(f"Run Command: {message['command']}")                     # log to history that the sent command was run
            exec(message["command"])                                               # run python command that was sent
        except Exception as e:
            logger.error("\n"+traceback.format_exc())
        _updateIot(device_name, mb )                                               # update iot state, in case schedule/status changed
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

def get_history( device_name ):
    """Get a list of scheduled commands from a device's shadow. This is intended for simple use cases"""
    warnings.filterwarnings("ignore")
    mb = messaging.MessageBroker(str(uuid.uuid4))                        # spin up iot
    status = mb.get_device_state( device_name )["history"]                # get schedule for device
    mb.shutdown()                                                        # shutdown iot
    return status                                                        # return schedule to user

def get_info( device_name ): 
    """Get public device info from its shadow. This is intended for simple use cases"""
    warnings.filterwarnings("ignore")
    mb = messaging.MessageBroker(str(uuid.uuid4))                             # spin up iot
    info = mb.get_device_state( device_name )                                 # get all info
    mb.shutdown()                                                             # shutdown iot
    return info                                                               # return info to user

def draw_schedule( device_list ):
    """Draw a weekly schedule of all events that occure for a chose device or experiment"""
    # To Do: figure out how to find all device if given the experiment.
    fig, ax = plt.subplots( figsize=(18, 15) )
    plt.title('Weekly Schedule', y=1, fontsize=16)       # Give the figure a title
    ax.grid(axis='y', linestyle='--', linewidth=0.5)     # Add horizonal grid lines to the plot

    DAYS = ['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ax.set_xlim(0.5, len(DAYS) + 0.5)
    ax.set_xticks(range(1, len(DAYS) + 1))
    ax.set_xticklabels(DAYS)
    ax.set_ylim( 24, 0)
    ax.set_yticks(range(0, 24))
    ax.set_yticklabels(["{0}:00".format(h) for h in range(0, 24)])

    # !!! This code should be able to handle experiment types 
    # Locally: schedule_str= [f"Job {i}: "+x.__repr__() for i,x in enumerate(schedule.get_jobs())]  #jobs= [job.__str__() for job in schedule.get_jobs()] 
    if type(device_list)==type(""):                    # if single device was passed to the function as a string instead of list, 
        device_list = [device_list]                    # turn it into a list
    colors= ["cornflowerblue","darkorange","mediumpurple","lightgreen"]
    legend_elements = [ Patch(colors[i],colors[i],alpha=0.3) for i in range(len(device_list)) ]
    plt.legend(legend_elements, device_list, bbox_to_anchor=(1.1,1), loc="upper right", prop={'size': 12})       # Add legend  # Show plot
    
    for dev_num, device in enumerate(device_list):
        info = get_info( device )
        jobs, schedule_str = info["jobs"], info["schedule"]

        for i in range(len(jobs)):                             # for each job, we get it's next run time, and run interval for IoT state information
            next_run= datetime.fromisoformat( schedule_str[i].split("next run: ")[1].split(")")[0] )
            period = timedelta(**{ jobs[i].split("unit=")[1].split(",")[0] : int(jobs[i].split("interval=")[1].split(",")[0]) })
            if "UNTIL-" in jobs[i]:                                                           # if there is a specified stop time, use it
                stop_time = datetime.fromisoformat( jobs[i].split("UNTIL-")[1] )
            else:                                                                             # otherwise stop at weekly cycle
                stop_time =  datetime.now() - timedelta(hours=7) + timedelta(weeks=1)     # Consider jobs that occur up to a week from now
            job_times = []                                                                 # create a list of all event times for a job
            while next_run < stop_time:
                job_times.append(next_run)
                next_run += period

            for event in job_times:
                d = event.weekday() + 0.52                                                   # get day of week for event
                start = float(event.hour) + float(event.minute) / 60                         # get start time of event
                end = float(event.hour) + (float(event.minute)+15) / 60                      # Ends 15 minutes after start
                plt.fill_between([d, d + 0.96], [start, start], [end, end], color=colors[dev_num],alpha=0.3)
                plt.text(d + 0.02, start + 0.02, '{0}:{1:0>2}'.format(event.hour, event.minute), va='top', fontsize=8)
                plt.text(d + 0.48, start + 0.01, f"Job {i}", va='top', fontsize=8) #ha='center', va='center', fontsize=10)
                
    # Add a red line for when right now is
    now =  datetime.now() - timedelta(hours=7) 
    d = now.weekday() + 0.52                                                   # get day of week for event
    start = float(now.hour) + float(now.minute) / 60                         # get start time of event
    end = float(now.hour) + (float(now.minute)+5) / 60                      # Ends 15 minutes after start
    plt.fill_between([d, d + 0.96 ], [start, start], [end, end], color='red' )
    plt.show()

    
def shutdown( device_name, hard=False ):
    """Stops iot listener on device by changing flag on shadow. This is intended for simple use cases"""
    send(device_name, "global iot_status; iot_status='shutdown'")
    if hard == True:
        mb = messaging.MessageBroker(str(uuid.uuid4))               # spin up iot
        mb.update_device_state( device_name, {"status":"shutdown"} )              # change status flag on device state to run
        mb.shutdown()   
    
def pause( device_name ):
    """Pauses iot listener on device by changing flag on shadow. This is intended for simple use cases"""
    send(device_name, "global iot_status; iot_status='pause'")  
    
def run( device_name ):
    """Resumes running of iot listener on device by changing flag on shadow. This is intended for simple use cases"""
    send(device_name, "global iot_status; iot_status='run'")
