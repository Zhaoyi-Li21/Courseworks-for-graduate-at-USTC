content_path = "/data2/home/zhaoyi/labs/USTC-labs/deeplearn_lab4/datasets/citeseer/citeseer.content"
cites_path = "/data2/home/zhaoyi/labs/USTC-labs/deeplearn_lab4/datasets/citeseer/citeseer.cites"
new_content_path = "/data2/home/zhaoyi/labs/USTC-labs/deeplearn_lab4/datasets/citeseer/citeseer.content.new"
new_cites_path = "/data2/home/zhaoyi/labs/USTC-labs/deeplearn_lab4/datasets/citeseer/citeseer.cites.new"
fr_content = open(content_path, "r")
fr_cites = open(cites_path, "r")
fw_content = open(new_content_path, "w")
fw_cites = open(new_cites_path, "w")

name_dict = dict()
ban_set = set()
content_lines = fr_content.readlines()
cites_lines = fr_cites.readlines()

import random

for line in content_lines:
    line = line.strip().split('	')
    name = line[0]
    try:
        x = int(name)
        if x in ban_set:
            print(x)
        ban_set.add(x)
    except ValueError:
        pass

for line in content_lines:
    line = line.strip().split('	')
    name = line[0]
    try:
        int(name)
        continue
    except ValueError:
        pass
    
    encode = random.randint(0, 200000)
    while encode in ban_set:
        encode = random.randint(0, 200000)
    ban_set.add(encode)
    name_dict[name] = str(encode)

for line in content_lines:
    temp = line
    line = line.strip().split('	')
    name = line[0]
    try:
        int(name)
        fw_content.write(temp)
    except ValueError:
        temp = temp.replace(name, name_dict[name])
        fw_content.write(temp)

for line in cites_lines:
    line = line.strip().split('	')
    n1 = line[0]
    n2 = line[1]
    try:
        int(n1)
        if n1 not in ban_set:
            continue
        n1 = str(n1)
    except ValueError:
        if n1 not in name_dict:
            continue
        n1 = name_dict[n1]
    
    try:
        int(n2)
        if n2 not in ban_set:
            continue
        n2 = str(n2)
    except ValueError:
        if n2 not in name_dict:
            continue
        n2 = name_dict[n2]

    fw_cites.write(n1+'	'+n2+'\n')
 
    
    

