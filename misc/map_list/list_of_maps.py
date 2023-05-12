import os, fnmatch


def find(extension, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, extension):
                temp = os.path.join(root, name).split('\\')
                result.append(temp[-1])
    return result

def  isFounded(extension, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, extension):
                temp = os.path.join(root, name).split('\\')
                result.append(temp[-1])
    if len(result) == 0:
        return False
    else:
        return True

#print('Set the path to the map folder: ')
#path = input()
#all_maps = find('*.SC2Map', path)
#map_count = len([map for map in all_maps if map != 0])
#if map_count == 0:
#    print('In this directory no maps, try again')
#else:
#    print(map_count)
#print(all_maps)