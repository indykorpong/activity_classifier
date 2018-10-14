import numpy as np
import math

def calculate_rval(a):
    a = int(a)
    r_val = -1.5*g + (a/63)*3*g      # Calculate real value from coded data
    r_val = float("{0:.8f}".format(r_val))

    return r_val

def calculate_si(data_arr):
    std_arr = []
    np_arr = np.array(data_arr)
    std_axis = np.std(np_arr,axis=0,ddof=1)
    
    std_arr.append(std_axis)
    std = np.array(std_arr)
    si_square = np.sum(std**2,axis=1)
    si_square = si_square/3
   
    return si_square

def calculate_si_m(data_arr):
    np_arr = np.array(data_arr)
    std_axis = np.std(np_arr,axis=0,ddof=1)
    std_axis = np.array(std_axis**2)

    return std_axis

filename = ["Dataset/Accelerometer-2011-04-11-13-28-18-brush_teeth-f1.txt",
            "Dataset/Accelerometer-2011-05-30-10-34-16-brush_teeth-m1.txt",
            "Dataset/Accelerometer-2011-05-30-21-55-04-brush_teeth-m2.txt",
            "Dataset/Accelerometer-2011-03-24-09-51-07-walk-f1.txt",
            "Dataset/Accelerometer-2011-05-30-21-44-35-walk-m2_1.txt",
            "Dataset/Accelerometer-2012-05-30-19-09-54-walk-m4.txt",
            "Dataset/Accelerometer-2011-03-24-10-04-32-pour_water-f1.txt",
            "Dataset/Accelerometer-2011-03-24-10-24-39-climb_stairs-f1.txt",
            "Dataset/Accelerometer-2011-05-30-10-20-47-climb_stairs-m1.txt",
            "Dataset/Accelerometer-2011-05-30-21-36-58-climb_stairs-m2.txt"]
std_arr = []
si_sq_arr = []

# Find the value of variance in each participant's acceleration data
# or Sigma_i_square

for f in filename:
    file = open(f,"r")
    # print(file.name)
    g = 9.8  # Gravitational constant
    count = 0
    data_arr = []
    
    for line in file:
        s = line.split(' ')
        temp = []
        for a in s:
            r_val = calculate_rval(a)
            temp.append(r_val)
        data_arr.append(temp)
        count += 1
        
    si_square = calculate_si(data_arr)
    # print(si_square)
    si_sq_arr.append(si_square[0])

# Find activity index value from Si_m when m is an acceleration axis number

file_count = 0
ai_all_act = []

for f in filename:
    print("#######################")
    print(" ")

    file = open(f,"r")
    print(file.name)
    print("=======================")

    g = 9.8  # Gravitational constant
    window_size = 10

    count = 0
    idx = 0

    data_arr = []
    sim_arr = []
    ai_arr = []

    for line in file:
        if(count==window_size):   # Calculate each window for every 200 timestamps
            
            sum_std_sq = 0  # Sum of Sigma_im_square

            count = 0
            idx += 1
            si_m = calculate_si_m(data_arr)
            for i in range(3):
                sum_std_sq += si_m[i]
            
            current_si_sq = si_sq_arr[file_count]
            diff_std = (sum_std_sq - current_si_sq)/current_si_sq

            activity_idx = math.sqrt(max(diff_std/3,0))
            # print("Activity Index: ", activity_idx)
            # print("=======================")
        
            ai_arr.append(activity_idx)

            data_arr = []   # Clear data array for the next 200 timestamp

        s = line.split(' ')
        temp = []
        for a in s:
            r_val = calculate_rval(a)   # Real value from raw data
            temp.append(r_val)
        data_arr.append(temp)
        count += 1

    file_count += 1

    np_ai_act = np.array(ai_arr)
    print("Numpy: ", np_ai_act)
    print(" ")

    ai_all_act.append(ai_arr)

    # Write all AI to a text file
    filename_ai = "Dataset/AI_" + file.name.split("/")[1]
    afile = open(filename_ai,"w")
    count = 0       # A file contains only 15 AIs.
    for a_val in ai_arr:
        if(count==15):
            break
        afile.write(str(a_val) + "\n")
        count += 1



np_all_act = np.array(ai_all_act)
# print("All activity's AI: ", ai_all_act)


