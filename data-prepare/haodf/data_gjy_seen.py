import json
import os


files_list = list
for root, dirs, files in os.walk("/mnt/data_process_bert_gpt_kg/multi_turn_data/"):
    files_list = [i for i in files if "valid" not in i]
    print(files_list)
    print("############")
    break
load_dict = list()
for file in files_list:
    with open("/mnt/data_process_bert_gpt_kg/multi_turn_data/" + file, 'r', encoding='utf8') as f:
        load_dict_temp = json.load(f)
        print(len(load_dict_temp))
        load_dict.extend(load_dict_temp)

load_dict_new = []
cnt = 0
c = 0
for idx, i in enumerate(load_dict):
    if idx%1000:
        print(cnt)
    d = i[0]
    if d == {}:
        c += 1
        continue
    flag = ",".join(d["flag"])
    dpt = ",".join(d["dpt"])
    hdpt = ",".join(d["hdpt"])
    dise = ",".join(d["dise"])
    hdise = ",".join(d["hdise"])
    sym = ",".join(d["sym"])
    hsym = ",".join(d["hsym"])
    check = ",".join(d["check"])
    hcheck = ",".join(d["hcheck"])
    drug = ",".join(d["drug"])
    hdrug = ",".join(d["hdrug"])
    food_pos = ",".join(d["food_pos"])
    hfood_pos = ",".join(d["hfood_pos"])
    food_neg = ",".join(d["food_neg"])
    hfood_neg = ",".join(d["hfood_neg"])
    info = [hfood_neg,hfood_pos,hdrug,hcheck,hsym,hdise,hdpt,food_neg,food_pos,drug,check,sym,dise,dpt]
    info = ",".join([i for i in info if i])
    temp_list = ["病人：" + info+"[SEP]"+ ",".join(i[2].split(",")[2:])+","+i[3][3:]] + [i[4]]
    if len(i[1]) > 1:
        for idx,j in enumerate(i[1][1:]):
            if "如果" in j["food_pos"]:
                j["food_pos"].remove("如果")
            if "结果" in j["food_pos"]:
                j["food_pos"].remove("结果")
            j_sym = ",".join(j["sym"])
            j_hsym = ",".join(j["hsym"])
            j_check = ",".join(j["check"])
            j_hcheck = ",".join(j["hcheck"])
            j_drug = ",".join(j["drug"])
            j_hdrug = ",".join(j["hdrug"])
            j_food_pos = ",".join(j["food_pos"])
            j_hfood_pos = ",".join(j["hfood_pos"])
            j_food_neg = ",".join(j["food_neg"])
            j_hfood_neg = ",".join(j["hfood_neg"])
            if flag == "disease":
                info = [j_hfood_neg,j_hfood_pos,j_hdrug,j_hcheck,j_hsym,j_food_neg,j_food_pos,j_drug,j_check,j_sym,dise,dpt]
            else:
                info = [j_hfood_neg,j_hfood_pos,j_hdrug,j_hcheck,j_hsym,j_food_neg,j_food_pos,j_drug,j_check,j_sym,sym,dpt]
            info = ",".join([i for i in info if i])
            temp_list.extend(["病人：" + info+"[SEP]" + ","+ i[5+idx][3:]] + [i[6+idx]])
    load_dict_new.append(temp_list)
    cnt += 1

with open("./data_multi/validate_lk_kg_info.json", 'w', encoding='utf-8') as file:
    file.write(json.dumps(load_dict_new, indent=2, ensure_ascii=False))
print(c)
