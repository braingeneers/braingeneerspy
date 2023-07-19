    
# import stuff
import uuid
import time
import warnings
from braingeneers.iot import messaging
from datetime import date, datetime, timedelta
from matplotlib.patches import Patch
import matplotlib.pyplot as plt


def send( device_name, command ):
    """Send a python script as a string which is then implemented by an IoT device. This is intended for simple use cases"""
    warnings.filterwarnings("ignore")
    my_id =str(uuid.uuid4() )
    mb = messaging.MessageBroker(my_id)                             # spin up iot
    mb.publish_message( topic=f"devices/dummy/{device_name}", message={"command": command,"id":my_id} )    # send command to listening device
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
