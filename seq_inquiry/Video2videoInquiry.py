import random
import data_seq
from scipy.spatial.distance import euclidean
from dtw import dtw
import numpy as np
from pylab import *


############################################ 
        #  Basic Utility Functions
############################################ 
############################################ 

def get_dist(vec_i, vec_j):
    return np.linalg.norm(vec_i - vec_j, ord=1)

    
def sum_dist(vecs_i, vecs_j):
    assert len(vecs_i) == len(vecs_j)
    _sum = 0
    for i in range(0, len(vecs_i)):
        _sum += get_dist( vecs_i[i], vecs_j[i])
    return _sum

    
def get_next_ij(dist, i , j):
    moves = [(0,-1), (-1,0), (-1,-1)]
    min_value = np.inf
    best_inci = -1 if i >0 else 0
    best_incj = -1 if j >0 else 0
    for (inci, incj) in moves:
        if i+inci >= 0 and j+incj >=0 and dist[i+inci, j+incj] < min_value:
            (best_inci, best_incj) = (inci, incj)
            min_value = dist[i+inci, j+incj]
    return (i+best_inci, j+best_incj)
        
def getBestDTWRoute(dist):
    (i,j) = dist.shape
    i=i-1
    j=j-1
    route = []
    path_i = []
    path_j = []
    while i!=0 or j!=0:
        route.append( (i,j) )
        path_i.append(i)
        path_j.append(j)
        i_next, j_next = get_next_ij(dist,i,j)
        (i,j) = (i_next, j_next)
#     for combos in route:
#         print(combos, "->")
#     path_i = path_i.reverse()
#     path_j = path_j.reverse()
    return (np.array(path_i), np.array(path_j))
        
def DTWdistance(s1,s2):
    n1 = len(s1)
    n2 = len(s2)
    dist = np.zeros((n1,n2))
    cost = np.zeros((n1,n2))
    dist[:,0] = np.inf
    dist[0,:] = np.inf
    dist[0,0] = 0
#     print("n1 n2", n1, n2)
    for i in range(0,n1):
        for j in range(0,n2):
#             print(i,j)
            cost[i,j] = get_dist(s1[i], s2[j])
            dist[i,j] = cost[i,j] + min(dist[i-1,j], dist[i,j-1], dist[i-1,j-1])
    path = getBestDTWRoute(dist)
    return dist[n1-1, n2-1], cost, dist, path

def NaiveMatch(cost):
    (n,m) = cost.shape
    min_sum = np.inf
    assert n<=m
    for startj in range(0, m-n+1):
        _sum = 0
        starti = 0
        for pos in range(0, n):
            _sum += cost[starti+pos, startj+pos]
        if _sum < min_sum:
            min_sum = _sum
    return min_sum
    
def get_category(str):
    return str.split('_')[1]


############################################ 
############################################ 
        #  Inquiry Database
############################################ 
############################################ 


def create_database(encoder, data,seq_length, class_limit, num_video_in_each_class):
    database = []
    N_database = class_limit * num_video_in_each_class
    for i in range(0, len(data.data)):
        smp = data.data[i]
        # ['train', 'BaseballPitch', 'v_BaseballPitch_g25_c06', '123']
        vid_len = int(smp[3])
        frams = data.get_frames_for_sample(smp,data.data_dir)
        seq = data.build_image_sequence(frams)
        seq = np.array(seq)
        seqX = []
        print('Idx:', "%d/%d"%(i, N_database),  smp[2], 'Length:' , len(seq)-seq_length,'Jump:', (seq_length>>1)+1 )
        for j in range(0,len(seq)-seq_length, (seq_length>>1)+1):
            seqX.append(seq[j:j+seq_length])
        seqX = np.array(seqX)
        seqY = encoder.predict(seqX)
        database.append((seqY, smp))
    return database

def initilize(encoder, data_path, seq_length, class_limit, num_video_in_each_class, random_class):
    # Initilize 
#     data_seq = reload(data_seq)
    import random
    data = data_seq.DataSet(data_dir = data_path, seq_length=seq_length,class_limit=class_limit, random_class=random_class)
    random.shuffle(data.data)
    print("Number of records in database:", len(data.data))
    # Get test set
    train, test = data.split_train_test()
    data.data = test
    # Filter number of videos in each class
    filter_list = []
    count_video_in_each_class = {}
    for i in test:
        _class = i[1]
        if _class not in count_video_in_each_class:
            count_video_in_each_class[_class] = 1
            filter_list.append(i)
        elif count_video_in_each_class[_class] < num_video_in_each_class:
            count_video_in_each_class[_class] += 1
            filter_list.append(i)
    data.data = filter_list
    print("Number of records in filtered database:", len(data.data))
    return data


def get_inquiry(encoder, smp, data, inq_length, config):
    
    vid_len = int(smp[3])
    frams = data.get_frames_for_sample(smp, data.data_dir)
    frams = data.rescale_list(frams, vid_len-1)
    seq = data.build_image_sequence(frams)
    seq = np.array(seq)
    
    seqX = []
    X_frames_start= []
    stepsize = vid_len / inq_length - 2
    for j in range(0, inq_length):
        jst = j*stepsize
        jend = j*stepsize+config.sequenceLength
        seqX.append(seq[jst:jend])
        X_frames_start.append((jst, jend))
    seqX = np.array(seqX)
    seqY = encoder.predict(seqX)
    
    return {"seqX": seqX,
            "seqY": seqY ,
            "smp": smp,
            "X_frames_start": X_frames_start}

def inquiry_in_database(encoder, data, database, config, inq_length , match_method = "dtw", Given_inquiry = False, inq_dict=None):
    
    if not Given_inquiry:
        index = np.random.choice(len(database))
        inq_smp = database[index][1]
        inq_dict = get_inquiry(encoder, inq_smp, data, inq_length, config)
        
    inquiry_seqY = inq_dict["seqY"]
            
        
#     if inq_length is not None:
#         seqY = []
#         seqY = np.array(seqY)
#         Y_frames_start= []
#         stepsize = len(inquiry_seqY) / inq_length - 2
#         for j in range(0, inq_length):
#             jst = j*stepsize
#             jend = j*stepsize+config.sequenceLength
#             if seqY.ndim == 1:
#                 seqY = inquiry_seqY[jst:jend]
#             else:
#                 seqY = np.concatenate([seqY, inquiry_seqY[jst:jend]])
#             Y_frames_start.append((jst, jend))
#         inquiry_seqY = seqY
    
    scores= []
    score_names = []
    bestscore = 1<<30
#     dist, cost, acc, path
    for (idx,i) in enumerate(database):
        if database[idx][1][2] == inq_dict["smp"][2]:
            continue

#         print(inquiry_seqY.shape) 8 * 14470
#         print(i[0].shape) ~50 * 14470

        seqyflat = inquiry_seqY.reshape((inquiry_seqY.shape[0], -1))
        iyflat = i[0].reshape((i[0].shape[0], -1))
        dist, cost, acc, path = DTWdistance(seqyflat, iyflat)            
        if match_method == "dtw":
            dist = dist
        elif match_method == "naive":
            # slide the inqury sequence over the candidate video to find the minimal match position
            dist = NaiveMatch(cost)
            
        scores.append((dist,i[1][2]))
        if dist< bestscore:
            bestscore = dist
            bestpath = path
            bestacc= acc
            bestfilename = i[1][2]
            
    scores = sorted(scores, key=lambda x:x[0])
    
    res_dict = {}
    res_dict["bestscore"] = bestscore
    res_dict["bestacc"] = bestacc
    res_dict["bestpath"] = bestpath
    res_dict["bestfilename"] = bestfilename
    res_dict["scores"] = scores
    return inq_dict, res_dict

def show_inquriy_stats(inq_dict, inq_result, show_top_limit = 10):
    print("Inqury is:\t"+ inq_dict["smp"][2])
    print("Inqury frames:\t"+ str(inq_dict["X_frames_start"]))
    print("Best Match is:\t"+ inq_result["bestfilename"])
    print("Best Dist:\t"+ str(inq_result["bestscore"]))

    acc = inq_result["bestacc"]
    path = inq_result["bestpath"]
    imshow(acc.T, origin='lower', cmap=cm.gray, interpolation='nearest')
    plot(path[0], path[1], 'w')
    # xlim((-0.5, acc.shape[0]-0.5))
    # ylim((-0.5, acc.shape[1]-0.5))

    print("========================")
    print("All scores in database:")
    for i in range( min(show_top_limit, len(inq_result["scores"])) ):
        print("Record: "+ inq_result["scores"][i][1].ljust(25)+ "\tDTW Dist: "+ str(inq_result["scores"][i][0]))
        
    




def multiple_test(data,run_times=100, if_itself=True):
    # how many top result will be counted
    Count_tops = 5
    
    Total_run = run_times
    # Tops_same_count: number of same category as inq appeared in result list
    Tops_same_count = [0.0] * Count_tops

    # Tops_same_ever_hit: if the same category as inq has appeared in result list of top N
    Tops_same_ever_hit = [0.0] * Count_tops
    
    # Sum of Score of Nth element of return list
    Nth_score_sum = [0.0] * Count_tops
    Hit_itself_sum = [0.0] * Count_tops
    self_pos_sum = 0
    
    run_times = min(run_times, len(data.data)-N_database)

    if if_itself:
        stpos = 0
    else:
        stpos = N_database
#     print(stpos)
    
    for i in range(0, run_times):
        if i %1 == 0:
            print("%d/%d Runs."%(i, run_times))
            
        inqs = get_inquiry(data, if_random = False, pos= stpos+i)
        seqY = inqs[1]
        inq_result = inquiry_in_database(seqY, database)
        for j in range(0, Count_tops):
            if(get_category(inq_result["scores"][j][1]) == get_category(inqs[2])):
                for k in range(j, Count_tops):
                    Tops_same_count[k] += 1
#                 break

        for j in range(0, Count_tops):
            if(get_category(inq_result["scores"][j][1]) == get_category(inqs[2])):
                for k in range(j, Count_tops):
                    Tops_same_ever_hit[k] += 1
                break
            
                
        for j in range(0, Count_tops):
            Nth_score_sum[j] += inq_result["scores"][j][0]
            if(inq_result["scores"][j][1] == inqs[2]):
                Hit_itself_sum[j] += 1
            
    top_cat_same = [x/Total_run for x in Tops_same_count]
    top_cat_same_hit = [x/Total_run for x in Tops_same_ever_hit]
    Nth_score_avg = [x/Total_run for x in Nth_score_sum]
    Hit_itself_avg = [x/Total_run for x in Hit_itself_sum]
    
    res_dict = {}
    res_dict["top_cat_same"] = top_cat_same
    res_dict["top_cat_same_hit"] = top_cat_same_hit
    res_dict["Nth_score_avg"] = Nth_score_avg
    res_dict["Hit_itself_avg"] = Hit_itself_avg
    
    print("top_cat_same: ", top_cat_same)
    print("top_cat_same_hit: ", top_cat_same_hit)
    print("Nth_score_avg: ", Nth_score_avg)
    print("Hit_itself_avg: ", Hit_itself_avg)
    
    return res_dict
