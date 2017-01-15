import math
import time
import sys

#       Note: Please use the road segments included in folder, if in case weird output comes because of /r in Linux files
#
#		We haven't included routes with speed limit 0 or " ", just considered that the routes are under maintenance
# 1.    For distance A Star works much better than the rest specially for long routes, DFS is the worst
#       For segments A Star works best but even BFS gives equally good results and is comparable to A Star
#       Given the time taken  by BFS to solve the path, BFS looks better than A Star(if sub optimal ans can be accepted)
#       For time A Star is far better than the rest, DFS just gives a route no one would like to follow
#       For scenic A Star and BFS gives good results, A Star being the best

# 2.    Results printed at the bottom
#       As seen from the results, DFS is best when compared w.r.t time complexity. Ratio depends on the pair of cities
#       we are searching for, for small routes A Star is almost comparable to dfs but for long routes
#       DFS leaves everyone behind by a big margin. On an average DFS is much better than the rest(2 times better
#       than BFS and 4 times better than A Star)

# 3.    Results printed at the bottom
#       IDS is comparatively better. However, there are times when A Star is better in terms of space complexity
#       To compare the results, we used th fringe length and the length of the city traverse tree. If we just take
#       fringe length, IDS seems to be better for most cases.

# 4.    I used c + h(s) function where c is the cost to reach the current city, h(s) is the admissible approximation of
#       the cost to reach destination from current city. The fringe is sorted based on c + h(s) function
#       and I am popping the 1st element each time
#
#       For distance -> c = distance to reach the current node, h(s) = distance between the latitude longitude between
#                    the current city and destination. If latitude longitude or current/destination are not present,
#                    I am finding the nearest city from that city/highway to get approximate area(find_lat_lon function)
#       For time     -> c = time to reach current node, h(s) = minimum time required to reach destination(distance
#                    between the latitude longitude between the current city and destination/highest speed limit)
#       For segments -> c = number of segments to reach current location, h(s) = minimum no. of segments required to
#                    reach destination(distance between the latitude longitude between the current city and
#                    destination/longest route in given data)
#       For scenic   -> Used combination of distance and speed limit. If speed limit is >55miles/hr just add the city to
#                    the end of fringe else insert based on c + h(s) which is based on distance)
#
#       If we find a already visited node in my successor function whose cost is less than the current current cost. I
#       am updating the parent for that node and iteratively updating other nodes which might be already visited and are
#       child of already updated nodes. This part is only done when we have already have found a path. If the time taken
#       to update nodes is greater than 50 sec. The sub optimal path is printed.
#
#       We can improve the heuristic function by making it run unless the cost of reaching a city in fringe is greater
#       than reaching the goal for the best route found till now. But this will increase time complexity by multiple
#       folds and so, we haven't implemented than and are returning the first result found.

# 5.    Skagway,_Alaska 4617 miles (a_star result), as the lengths given in Canada are in kms and not in miles
#       we are getting the result which is more than the actual path length we get using a_star.


def read_road_segments():               # Read road segments into a dictionary
    file1 = open('road-segments.txt', 'r')
    result = {}
    longest_route_length = 0
    slowest_speed = 100
    fastest_speed = 0
    for line in file1.readlines():

        city_data_on_a_line = line.split(" ")
        city_data_on_a_line[4] = city_data_on_a_line[4][:-2]
        if (not city_data_on_a_line[3] == '') and not (int(city_data_on_a_line[3]) == 0):
            # if the speed limit is 0, we are considering that the road is under construction or is blocked
            # because of the ongoing maintenance
            if city_data_on_a_line[0] in result:
                result[city_data_on_a_line[0]]['connected_cities'] += [city_data_on_a_line[1]]
                result[city_data_on_a_line[0]][city_data_on_a_line[1]] = [int(city_data_on_a_line[2]),
                                                                          int(city_data_on_a_line[3]), float(
                        float(city_data_on_a_line[2]) / float(city_data_on_a_line[3]) * 60.0),
                                                                          city_data_on_a_line[4]]
                if longest_route_length < int(city_data_on_a_line[2]):
                    longest_route_length = int(city_data_on_a_line[2])
                if slowest_speed > int(city_data_on_a_line[3]):
                    slowest_speed = int(city_data_on_a_line[3])
                if fastest_speed < int(city_data_on_a_line[3]):
                    fastest_speed = int(city_data_on_a_line[3])
            else:
                result[city_data_on_a_line[0]] = {'connected_cities': [city_data_on_a_line[1]]}
                result[city_data_on_a_line[0]][city_data_on_a_line[1]] = [int(city_data_on_a_line[2]),
                                                                          int(city_data_on_a_line[3]), float(
                        float(city_data_on_a_line[2]) / float(city_data_on_a_line[3]) * 60.0),
                                                                          city_data_on_a_line[4]]

            if city_data_on_a_line[1] in result:
                result[city_data_on_a_line[1]]['connected_cities'] += [city_data_on_a_line[0]]
                result[city_data_on_a_line[1]][city_data_on_a_line[0]] = [int(city_data_on_a_line[2]),
                                                                          int(city_data_on_a_line[3]), float(
                        float(city_data_on_a_line[2]) / float(city_data_on_a_line[3]) * 60.0),
                                                                          city_data_on_a_line[4]]

            else:
                result[city_data_on_a_line[1]] = {'connected_cities': [city_data_on_a_line[0]]}
                result[city_data_on_a_line[1]][city_data_on_a_line[0]] = [int(city_data_on_a_line[2]),
                                                                          int(city_data_on_a_line[3]), float(
                        float(city_data_on_a_line[2]) / float(city_data_on_a_line[3]) * 60.0),
                                                                          city_data_on_a_line[4]]

    file1.close()
    return [result, longest_route_length, slowest_speed, fastest_speed]


def merge_sort(city, start_point, end_point, routing_parameter):    # Sorting connected city list of the given city(ordering input for bfs/dfs/ids)
    connected_city = road_segments[city]["connected_cities"]
    if start_point == end_point:
        return [connected_city[start_point]]
    else:
        l1 = merge_sort(city, start_point, int(math.floor((start_point + end_point) / 2)), routing_parameter)
        l2 = merge_sort(city, int(math.floor((start_point + end_point) / 2) + 1), end_point, routing_parameter)

        result = merge(city, l1, l2, routing_parameter)
        if routing_parameter == "segments":
            result.reverse()
        return result


def merge(city, l1, l2, routing_algo_option):       # Merge 2 sorted list
    result = []
    if routing_algo_option == "distance":
        index = 0
    elif routing_algo_option == "time":
        index = 2
    elif routing_algo_option == "scenic":
        index = 1
    elif routing_algo_option == "segments":
        index = 0

    while len(l1) > 0 and len(l2) > 0:
        if road_segments[city][l1[0]][index] < road_segments[city][l2[0]][index]:
            result.append(l1[0])
            l1.pop(0)
        else:
            result.append(l2[0])
            l2.pop(0)

    if len(l1) > 0:
        result += l1
    elif len(l2) > 0:
        result += l2

    return result


def read_city_gps():            # Reading city gps into a dictionary
    file1 = open('city-gps.txt', 'r')
    result = {}
    for line in file1.readlines():
        city_data_on_a_line = line.split(" ")
        result[city_data_on_a_line[0]] = [float(city_data_on_a_line[1]),
                                          float(city_data_on_a_line[2].split("\n")[0][:-1])]
    file1.close()
    return result


def distance_lat_lon(latitude1, longitude1, latitude2, longitude2):     # Distance in miles between 2 lat lon
    lat1_radians = math.radians(latitude1)
    lat2_radians = math.radians(latitude2)

    diff_lat = math.radians(abs(latitude2 - latitude1))
    diff_lon = math.radians(abs(longitude2 - longitude1))
    a = (math.pow((math.sin(diff_lat / 2.0)), 2)) + math.cos(lat1_radians) * math.cos(lat2_radians) * (
        math.pow((math.sin(diff_lon / 2.0)), 2))
    c = 2.0 * (math.atan2(*(math.sqrt(a), math.sqrt(1 - a))))
    # d = 6373*c  #in km
    d = 3960 * c  # in miles
    return d


def successors_cities(city, parameter, routing_algo):
    # list of successor cities. It not only passes the neighbour city but also
    #  current path length(may be in time/distance/segments)
    result = []
    if parameter == "distance":
        index = 0
    elif parameter == "time":
        index = 2
    elif parameter == "scenic":
        index = 1
    else:           # "segments":
        index = 0

    if routing_algo != "astar":
        road_segments[city[0]]["connected_cities"] = merge_sort(city[0], 0,
                                                                len(road_segments[city[0]]["connected_cities"]) - 1,
                                                                parameter)

        if routing_algo == "dfs" or routing_algo == "ids":
            road_segments[city[0]]["connected_cities"].reverse()
        for neighbour_city in road_segments[city[0]]["connected_cities"]:
            result.append([neighbour_city, city[0], (city[2] + 1)])

    else:
        for neighbour_city in road_segments[city[0]]["connected_cities"]:
            if parameter == "scenic":
                    result.append([neighbour_city, city[0], city[2], road_segments[city[0]][neighbour_city][index]])
            elif parameter == "segments":
                result.append([neighbour_city, city[0], city[2] + 1])
            else:
                result.append([neighbour_city, city[0],
                           city[2] + road_segments[city[0]][neighbour_city][index]])
    return result


def solve_dfs_bfs(start_city, end_city, routing_option, routing_algorithm, depth): # Basic dfs/bfs/ids
    fringe = [[start_city, 'root', 0]]
    visited_cities = [start_city]
    city_traverse_list_tree = {start_city: [start_city, 'root', 0]}
    while len(fringe) > 0:

        if routing_algorithm == "bfs":
            current_city = fringe.pop(0)
        else:  # dfs or ids
            current_city = fringe.pop()

        if routing_algorithm == "ids":
            if current_city[2] >= depth:
                continue
        if current_city == end_city:
            # print len(fringe)
            return city_traverse_list_tree

        for s in successors_cities(current_city, routing_option, routing_algorithm):
            if s[0] in visited_cities:
                continue

            visited_cities.append(s[0])
            city_traverse_list_tree[s[0]] = s
            fringe.append(s)
            if s[0] == end_city:
                # print len(fringe)
                return city_traverse_list_tree
    return False


def solve_ids(start_city, end_city, routing_option, routing_algorithm):
    # Reusing bfs_dfs functions to implement ids
    # Calling bfs_dfs function and incrementing depth in each call
    depth = 0
    while True:
        result = solve_dfs_bfs(start_city, end_city, routing_option, routing_algorithm, depth)
        if result:
            return result
        depth += 1


def solve_a_star(start_city, end_city, routing_option, optimal):
    # Uses c + h(s) for determining the best successor
    fringe = [start_city]
    visited_cities = [start_city]
    city_traverse_list_tree = {start_city: [start_city, 'root', 0]}
    # A dictionary maintaining city, parent, cost to current node(cost can be in the form of distance, time, segments)
    while len(fringe) > 0:
        # As fringe is sorted by cost, we are just popping first element
        current_city = fringe.pop(0)
        for s in successors_cities(city_traverse_list_tree[current_city], routing_option, "astar"):
            if s[0] in visited_cities:
                # When city in successor is in visited list, we check if the newly founded cost is less
                # than the already present cost. If so update parent iteratively checking if the newly updated cost
                # is less than existing cost.
                if city_traverse_list_tree[s[0]][2] > s[2] and optimal == True:
                    find_new_parent = find_new_path(s, city_traverse_list_tree, fringe, end_city, routing_option)
                    if not find_new_parent:
                        return False
                    city_traverse_list_tree = find_new_parent[0]
                    fringe = find_new_parent[1]
                continue
            visited_cities.append(s[0])
            city_traverse_list_tree[s[0]] = s

            if s[0] == end_city:
                # print len(fringe)
                return city_traverse_list_tree

            fringe = add_city_to_fringe(fringe, s[0], end_city, city_traverse_list_tree, routing_option)
    return False


def find_new_path(current_city_details, city_traverse_list_tree, fringe, end_city, routing_option):
    city_traverse_list_tree[current_city_details[0]] = current_city_details
    fringe = add_city_to_fringe(fringe, current_city_details[0], end_city, city_traverse_list_tree, routing_option)
    for connected_city in road_segments[current_city_details[0]]["connected_cities"]:
        if time.time() - start_time > 50:
            return False
        if connected_city in city_traverse_list_tree:
            if city_traverse_list_tree[connected_city][1] == current_city_details[0]:
                city_traverse_list_tree[connected_city][2] = current_city_details[2] + \
                                                             road_segments[connected_city][current_city_details[0]][1]
                find_new_path(city_traverse_list_tree[connected_city], city_traverse_list_tree, fringe, end_city, routing_option)

            elif city_traverse_list_tree[connected_city][2] > road_segments[connected_city][current_city_details[0]][
                1] + city_traverse_list_tree[current_city_details[0]][2]:
                city_traverse_list_tree[connected_city][1] = current_city_details[0]
                city_traverse_list_tree[connected_city][2] = road_segments[connected_city][current_city_details[0]][1] \
                                                             + city_traverse_list_tree[
                                                                 current_city_details[0]][2]
                find_new_path(city_traverse_list_tree[connected_city], city_traverse_list_tree, fringe, end_city, routing_option)

    return [city_traverse_list_tree, fringe]


def add_city_to_fringe(fringe, city, end_city, city_traverse_list_tree, routing_option):
    # add city to fringe such that the fringe remains sorted based on cost
    if city in fringe:
        fringe.remove(city)
    i = 0
    lat_lon_of_current_city = find_lat_lon_fixed(city)
    lat_lon_of_end_city = find_lat_lon_fixed(end_city)
    distance_between_current_city_end_city = distance_lat_lon(lat_lon_of_current_city[0],
                                                              lat_lon_of_current_city[1],
                                                              lat_lon_of_end_city[0], lat_lon_of_end_city[1])

    if city_traverse_list_tree[city][2] > 55 and routing_option == "scenic":
        fringe.append(city)
        return fringe

    for city_fringe in fringe:
        lat_lon_of_fringe_city = find_lat_lon_fixed(city_fringe)
        distance_between_city_fringe_end_city = distance_lat_lon(lat_lon_of_fringe_city[0],
                                                                 lat_lon_of_fringe_city[1],
                                                                 lat_lon_of_end_city[0], lat_lon_of_end_city[1])
        if routing_option == "distance":
            if distance_between_current_city_end_city + city_traverse_list_tree[city][2] \
                    < distance_between_city_fringe_end_city + city_traverse_list_tree[city_fringe][2]:
                break
        elif routing_option == "time":
            if distance_between_current_city_end_city/fastest_speed_global + city_traverse_list_tree[city][2] \
                    < distance_between_city_fringe_end_city/fastest_speed_global + city_traverse_list_tree[city_fringe][2]:
                break
        elif routing_option == "segments":
            if distance_between_current_city_end_city/longest_route_length_global + city_traverse_list_tree[city][2] \
                    < distance_between_city_fringe_end_city/longest_route_length_global + city_traverse_list_tree[city_fringe][2]:
                break
        elif routing_option == "scenic":
            if distance_between_current_city_end_city < distance_between_city_fringe_end_city:
                break
        i += 1
    if i >= len(fringe):
        fringe.append(city)
    else:
        fringe.insert(i, city)
    return fringe


def find_lat_lon(city, count):
    # Find lat lon of the given city. If lat lon not found, search for the nearest neighbour(approximate)
    if city in city_gps:
        return [city_gps[city][0], city_gps[city][1], city]
    if count > 3:
        return [0, 0]
    else:
        nearest_city_distance = 200000
        city_with_result_false = []
        while True:
            for neighbour_city in road_segments[city]["connected_cities"]:
                if neighbour_city in city_with_result_false:
                    continue
                if nearest_city_distance > road_segments[city][neighbour_city][0]:
                    nearest_city_distance = road_segments[city][neighbour_city][0]
                    nearest_city = neighbour_city
                    return find_lat_lon(nearest_city, count + 1)


def find_lat_lon_fixed(city):
    result = find_lat_lon(city, 0)
    if result[0] == 0 and result[1] == 0:
        for neighbour_city in road_segments[city]["connected_cities"]:
            if neighbour_city in city_gps:
                return find_lat_lon(neighbour_city, 0)
            return [0, 0]
    else:
        return result


def solve(start_city, end_city, routing_option, routing_algorithm):
    # Calling the right function
    city_traverse_tree = []
    if not (start_city in road_segments or end_city in road_segments):
        return False

    if start_city == end_city:
        city_traverse_tree = {start_city: [start_city, 'root', 0]}
    elif routing_algorithm == "dfs":
        city_traverse_tree = solve_dfs_bfs(start_city, end_city, routing_option, "dfs", 0)
    elif routing_algorithm == "bfs":
        city_traverse_tree = solve_dfs_bfs(start_city, end_city, routing_option, "bfs", 0)
    elif routing_algorithm == "ids":
        city_traverse_tree = solve_ids(start_city, end_city, routing_option, "ids")
    elif routing_algorithm == "astar":
        city_traverse_tree = solve_a_star(start_city, end_city, routing_option, False)
        if time.time() - start_time < 10:
            city_traverse_tree2 = solve_a_star(start_city, end_city, routing_option, True)
            if city_traverse_tree2 != False:
                city_traverse_tree = city_traverse_tree2
        # result = path_display(city_traverse_tree, end_city)
        # print result
        #print "Trying for a better optimal path. This might take time....."
        #print time.asctime(time.localtime(time.time()))
        #
        #result = path_display(city_traverse_tree, end_city)
        #print result
    #print len(city_traverse_tree)
    return city_traverse_tree


def find_lat_lon_path(path_list):
    # Return lat lon of the path found. Useful to map it on a site (Ex: http://www.darrinward.com/lat-long)
    # to check what path you have found based on lat, lon list
    result = ""
    for city in path_list:
        lat_lon = find_lat_lon_fixed(city)
        result += str(lat_lon[0]) + "," + str(lat_lon[1]) + '\n'
    return result


def path(traverse_tree, city):
    result = [city]
    reverse_result = []
    distance = 0
    route_time = 0
    segments = 0
    highways = 0

    while True:
        parent = traverse_tree[city][1]
        if parent == 'root':
            break
        segments += 1
        if road_segments[city][parent][1] > 55:
            highways += 1
        distance += road_segments[city][parent][0]
        route_time += road_segments[city][parent][2]
        result.append(parent)
        city = parent
    i = len(result)
    while i > 0:
        reverse_result += [result[i - 1]]
        i -= 1
    return [reverse_result, distance, route_time, segments, highways]


def path_display(traverse_tree, city):
    end_city = city
    result = [city]
    reverse_result = ""
    distance = 0
    route_time = 0

    result_print = []

    while True:
        parent = traverse_tree[city][1]
        if parent == 'root':
            break
        distance += road_segments[city][parent][0]
        route_time += road_segments[city][parent][2]
        result.append(parent)
        result_print.append("Go from " + str(parent) + " to " + str(city) + " via " + str(road_segments[city][parent][3]) + ", Speed Limit: " + str(road_segments[city][parent][1]) + "miles/hr, Distance: " + str(road_segments[city][parent][0]) + "miles")
        city = parent
    i = len(result)
    j = len(result_print)
    #print result_print
    result_string = ""
    while j > 0:
        result_string += str(result_print[j - 1])
        j -= 1
        if j != 0:
            result_string += "\nNext "
    print "\nStart city: " + city + ", End city: " + end_city + ", Route time: " + str(float(route_time)/60.0)[:5] + "hrs, Route distance: " + str(distance) + " miles\n"
    print result_string
    print "You reached your destination\n"
    while i > 0:
        reverse_result += str(result[i - 1]) + " "
        i -= 1
    return str(distance) + " " + str(float(route_time)/60.0) + " " + str(reverse_result)


def starts_here():
    if len(sys.argv) < 5:
        print "Less than expected no. of arguments"
    elif len(sys.argv) > 5:
        print "More than expected no. of arguments"
    else:

        """start_city_end_city_list = [['"Y"_City,_Arkansas', "Dover-Foxcroft,_Maine"],  # 0
                                    ['"Y"_City,_Arkansas', 'Forgan,_Oklahoma'],  # 1
                                    ["New_York,_New_York", "Los_Angeles,_California"],  # 2
                                    ["New_York,_New_York", "Chicago,_Illinois"],  # 3
                                    ["New_York,_New_York", "Miami,_Florida"],  # 4
                                    ["New_York,_New_York", "Skagway,_Alaska"],  # 5
                                    ["Miami,_Florida", "Skagway,_Alaska"],  # 6
                                    ["New_York,_New_York", "Weed,_California"],  # 7
                                    ["New_York,_New_York", "Newburgh,_New_York"],  # 8
                                    ["New_York,_New_York", 'Buffalo,_New_York'],  # 9
                                    ["New_York,_New_York", 'New_York,_New_York'],   # 10
                                    ["Matamoros,_Tamaulipas", "Mexicali,_Baja_California_Norte"],   # 11
                                    ["Miami,_Florida", "Tampa,_Florida"],   # 12
                                    ["Miami,_Florida", "New_York,_New_York"],    # 13
                                    ["Miami,_Florida", "Skagway,_Alaska"],     # 14
                                    ["Miami,_Florida", "Los_Angeles,_California"]]  # 15"""

        #path_between = start_city_end_city_list[3]
        routing_algo = ["dfs", "bfs", "ids", "astar"]
        routing_option = ["distance", "time", "scenic", "segments"]
        start_city_arg = sys.argv[1]
        end_city_arg = sys.argv[2]
        routing_algorithm_arg = sys.argv[4]
        routing_option_arg = sys.argv[3]

        if not(start_city_arg in road_segments and end_city_arg in road_segments):
            print "One or both of the cities not in out city directory"
        elif start_city_arg == end_city_arg:
            print "You are already in the end city\n0 0 " + start_city_arg + " " + end_city_arg
        elif routing_option_arg in routing_option and routing_algorithm_arg in routing_algo:
            traverse_tree = solve(start_city_arg, end_city_arg, routing_option_arg, routing_algorithm_arg)
            if traverse_tree:
                result = path_display(traverse_tree, end_city_arg)
                print result

                # result = path(traverse_tree, end_city_arg)
                # print result
                # print find_lat_lon_path(result[0]) # to print lat lon, used to map on a site
            else:
                print "Path not found"
        else:
            print "Invalid arguments"


# print time.asctime(time.localtime(time.time()))
start_time = time.time()

road_segments_result = read_road_segments()
city_gps = read_city_gps()
#print city_gps["New_York,_New_York"]
road_segments = road_segments_result[0]
#print road_segments["New_York,_New_York"]
longest_route_length_global = road_segments_result[1]
slowest_speed_global = road_segments_result[2]
fastest_speed_global = road_segments_result[3]

starts_here()

# end_time = time.time()

# print time.asctime(time.localtime(time.time()))
# print "Start to End " + str(end_time - start_time)

# DON'T DELETE
# try the below site to see route
# http://www.darrinward.com/lat-long


"""2. start-time	start-city end-city
a star
0.148000001907 "Y"_City,_Arkansas Dover-Foxcroft,_Maine
1.72000002861 "Y"_City,_Arkansas Forgan,_Oklahoma
1.73300004005 New_York,_New_York Los_Angeles,_California
4.27699995041 New_York,_New_York Chicago,_Illinois
4.35199999809 New_York,_New_York Miami,_Florida
4.51800012589 New_York,_New_York Skagway,_Alaska
7.4470000267 Miami,_Florida Skagway,_Alaska
10.6749999523 New_York,_New_York Weed,_California
13.2380001545 New_York,_New_York Newburgh,_New_York
13.2400000095 New_York,_New_York Buffalo,_New_York
13.256000042 New_York,_New_York New_York,_New_York
13.256000042 Matamoros,_Tamaulipas Mexicali,_Baja_California_Norte
13.2750000954 Miami,_Florida Tampa,_Florida
13.2769999504 Miami,_Florida New_York,_New_York
13.3500001431 Miami,_Florida Skagway,_Alaska
16.5989999771 Miami,_Florida Los_Angeles,_California
16.9630000591
dfs
0.194000005722 "Y"_City,_Arkansas Dover-Foxcroft,_Maine
0.984000205994 "Y"_City,_Arkansas Forgan,_Oklahoma
1.1970000267 New_York,_New_York Los_Angeles,_California
1.35000014305 New_York,_New_York Chicago,_Illinois
1.40700006485 New_York,_New_York Miami,_Florida
1.76700019836 New_York,_New_York Skagway,_Alaska
1.90300011635 Miami,_Florida Skagway,_Alaska
1.99700021744 New_York,_New_York Weed,_California
2.14100003242 New_York,_New_York Newburgh,_New_York
2.14300012589 New_York,_New_York Buffalo,_New_York
2.16300010681 New_York,_New_York New_York,_New_York
2.16400003433 Matamoros,_Tamaulipas Mexicali,_Baja_California_Norte
2.32899999619 Miami,_Florida Tampa,_Florida
2.55400013924 Miami,_Florida New_York,_New_York
3.53299999237 Miami,_Florida Skagway,_Alaska
3.625 Miami,_Florida Los_Angeles,_California
4.36100006104
bfs
0.146000146866 "Y"_City,_Arkansas Dover-Foxcroft,_Maine
1.46900010109 "Y"_City,_Arkansas Forgan,_Oklahoma
1.52400016785 New_York,_New_York Los_Angeles,_California
2.7610001564 New_York,_New_York Chicago,_Illinois
3.00300002098 New_York,_New_York Miami,_Florida
3.85500001907 New_York,_New_York Skagway,_Alaska
4.26900005341 Miami,_Florida Skagway,_Alaska
5.58500003815 New_York,_New_York Weed,_California
6.45700001717 New_York,_New_York Newburgh,_New_York
6.45900011063 New_York,_New_York Buffalo,_New_York
6.47500014305 New_York,_New_York New_York,_New_York
6.47500014305 Matamoros,_Tamaulipas Mexicali,_Baja_California_Norte
6.51800012589 Miami,_Florida Tampa,_Florida
6.51900005341 Miami,_Florida New_York,_New_York
6.90700006485 Miami,_Florida Skagway,_Alaska
8.21500015259 Miami,_Florida Los_Angeles,_California
9.24300003052"""

# 2.    20 runs
#       distance as the route option
#       algo time(in sec) fringe-length traverse-tree length(visited city, other info)
#       NY Skagway
#       bfs 7.5 141 3392
#       dfs 3.47 892 2372
#       astar 55 103 6367
#       ids 8.99*10 97 1449

#       50 runs
#       NY Chicago
#       ids 93.83*5 122 1431
#       bfs 24.04 114 2636
#       dfs 6.34 698 1533
#       astar 7.4 98 470

#       20 runs
#       NY LA
#       ids 76.62*10 293 5240
#       bfs 24.70 100 6137
#       dfs 3.898 977 2580
#       astar 50.06 307 4262

# 3.    distance as the route option
#       algo fringe-length traverse-tree-length(visited city, other info)
#       Bloomington Skagway
#       dfs 889 2095
#       a star 112 6336
#       ids 117 1946

#       Miami Skagway
#       ids 272 3763
#       astar 104 6366
#       bfs 46 6368
#       dfs 731 1943

#       Miami LA
#       astar 154 1340
#       ids 331 2528
#       bfs 113 5519
#       dfs 816 2151

#       Miami NY
#       astar 62 578
#       ids 84 693

#       Miami Tampa
#       ids 3 40
#       astar 11 40