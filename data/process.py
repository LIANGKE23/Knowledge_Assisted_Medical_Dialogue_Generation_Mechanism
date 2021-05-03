import re
import json


def processing(file_path):
    data = []
    f = open(file_path, "r", encoding="utf8")
    cnt = 1
    data_tmp = []
    dpt_flag = False
    desc_flag = False
    dialogue_flag = False
    patient = False
    doctor = False
    while True:
        line = f.readline()
        if not line:
            break
        if line.startswith("id=") and data_tmp:  # 新一轮开始，把之前保存的对话加入data
            data.append(["ID:" + str(cnt), "Turn:" + str(int((len(data_tmp) - 1) / 2))] + data_tmp)
            cnt += 1
            print(cnt)
            print(["ID:" + str(cnt), "Turn:" + str(int((len(data_tmp) - 1) / 2))] + data_tmp)
            data_tmp = []
            dialogue_flag = False
        elif line.startswith("Doctor faculty"):
            dpt_flag = True
        elif dpt_flag:
            if "医院" in line or "科" in line:
                dpt = line.strip("\n").split()[-1]
                if "（" in dpt:
                    dpt = line.strip("\n").split()[-1]
                data_tmp.append("科室：" + dpt)
            dpt_flag = False
        elif line.startswith("Description"):
            desc_flag = True
        elif desc_flag:
            if "2010" not in file_path:  # 2010年以外的数据需要把Description内容拼成一段text，方便进行后续操作
                while True:
                    line_tmp = f.readline()
                    if line_tmp == "\n" and "2012" in file_path:
                        line_tmp = f.readline()
                        if "Dialogue" in line_tmp:
                            dialogue_flag = True
                            break
                    elif line_tmp == "\n":
                        break
                    line += line_tmp
            str1, str2, str3, str4, str5 = [], [], [], [], []  # 找出 疾病和帮助后的内容，作为病人的第一次提问
            if "2010" in file_path:
                line = line.replace(":", "：")
                str1 = re.findall("疾病：(.*?) ", line)
                str2 = re.findall("帮助：(.*?)\n", line)
                str3 = re.findall("就诊医院等）：(.*?)想得到怎样的帮助", line)
                if not str3:
                    str3 = re.findall("发病时间）：(.*?)想得到怎样的帮助", line)
                if not str3:
                    str3 = re.findall("发病时间）：(.*?)想获得的帮助", line)
                if not str3:
                    str3 = re.findall("内容：(.*?)想获得的帮助", line)
                if not str3:
                    str3 = re.findall("内容：(.*?)想得到怎样的帮助", line)
                if str3:
                    str3 = [str3[0].replace("|", "")]
                    str3 = [str3[0].replace("曾经治疗情况和效果：", "|")]
            elif "2011" in file_path:
                line = line.replace(" ", "")
                str1 = re.findall("疾病：\n(.*?)\n", line)
                str2 = re.findall("帮助：\n(.*?)\n", line)
                str3 = re.findall("就诊医院等）：\n(.*?)\n", line)
                str4 = re.findall("曾经治疗情况和效果：\n(.*?)\n", line)
                if str3 and str4:
                    str3 = [str3[0].replace("|", "")]
                    str4 = [str4[0].replace("|", "")]
                    str3 = [str3[0] + "|" + str4[0]]
                else:
                    str3 = []
            elif "2018" in file_path or "2019" in file_path or "2020" in file_path:
                line = line.replace(" ", "")
                str1 = re.findall("疾病：\n(.*?)\n", line)
                str2 = re.findall("帮助：\n(.*?)\n", line)
                str3 = re.findall("病情描述：\n(.*?)\n", line)
                if str3:
                    str3 = [str3[0].replace("|", "")]
                else:
                    str3 = []
            elif "2012" in file_path:
                str1 = re.findall("疾病：\n(.*?)\n", line)
                str2 = re.findall("帮助：(.*?)\n", line)
                str3 = re.findall("治疗情况：\n(.*?)\n病史", line)
                str4 = re.findall("病史：\n(.*?)想得到怎样的帮助", line)
                if not str3 and not str4:
                    str5 = re.findall("就诊医院等）：(.*?)想得到怎样的帮助", line)
                    if str5:
                        str5 = [str5[0].replace("曾经治疗情况和效果：", "|")]
                    else:
                        str5 = []
                if str3 and str4:
                    str3 = [str3[0].replace("|", "")]
                    str4 = [str4[0].replace("|", "")]
                    str3 = [str3[0] + "|" + str4[0]]
                elif str5:
                    str3 = str5
                else:
                    str3 = []
            elif "2013" in file_path or "2014" in file_path or "2015" in file_path \
                    or "2016" in file_path or "2017" in file_path:
                str1 = re.findall("疾病：(.*?)\n", line)
                str2 = re.findall("帮助：(.*?)\n", line)
                str3 = re.findall("病情描述：(.*?)\n", line)
                if str3:
                    str3 = [str3[0].replace("|", "")]
                else:
                    str3 = []
            # str2 = re.findall("病情描述：(.*?)曾经治疗情况和效果", line)
            # if not str2:
            #     str2 = re.findall("）：(.*?)曾经治疗情况和效果", line)
            # if not str2:
            #     str2 = re.findall("内容：(.*?)曾经治疗情况和效果", line)
            # if not str2:
            #     str2 = re.findall("病情描述：(.*?)想得到怎样的帮助", line)
            # if not str2:
            #     str2 = re.findall("）：(.*?)想得到怎样的帮助", line)
            # if not str2:
            #     str2 = re.findall("内容：(.*?)想得到怎样的帮助", line)
            # if not str2:
            #     str2 = re.findall("病情描述：(.*?)现在病情、想获得的帮助", line)
            # if not str2:
            #     str2 = re.findall("）：(.*?)现在病情、想获得的帮助", line)
            # if not str2:
            #     str2 = re.findall("内容：(.*?)现在病情、想获得的帮助", line)
            if str1 and str2 and str3 and str1[0] and str2[0] and str3[0]:  # 默认每个case里都有疾病和帮助，没有的case略过吧
                str1, str2, str3 = str1[0], str2[0], str3[0]
                str1 = str1.replace(" ", "")
                str1 = str1.replace("|", "")
                str2 = str2.replace(" ", "")
                str2 = str2.replace("|", "")
                str3 = str3.replace(" ", "")
                str2 = str2.replace("化验、检查结果：", "。")
                if str2 and str2[-1] not in '.。？，,?!！~～、':
                    str2 += '。'
                if str3 and str3[-1] not in '.。？，,?!！~～、':
                    str3 += '。'
                if len(str3) > 3 and not str3.startswith("|"):
                    data_tmp.append("病人：" + str1 + "|" + str3 + "|病人：" + str2)
            desc_flag = False
        elif line.startswith("Dialogue"):
            dialogue_flag = True
        elif line.startswith("病人"):
            patient, doctor = True, False
        elif line.startswith("医生"):
            patient, doctor = False, True
        elif dialogue_flag:
            if len(data_tmp) < 2:  # 如果这个case没有科室数据掠过
                continue
            line = line.strip()
            line.replace(" ", "")
            if line and line[-1] not in '.。？，,?!！~～':
                line += '。'
            if not line:  # dialogue结束，第一句话不能是医生，最后一句话不能是病人
                if data_tmp[-1][:2] == "病人":
                    data_tmp = data_tmp[:-1]
                if len(data_tmp) > 1 and data_tmp[1][:2] == "医生":
                    data_tmp = data_tmp[:1]
                dialogue_flag = False
            elif (data_tmp[-1][:2] == "病人" and patient) or (data_tmp[-1][:2] == "医生" and doctor):
                data_tmp[-1] += line
            else:
                if patient:
                    data_tmp.append("病人：" + line)
                if doctor:
                    data_tmp.append("医生：" + line)
    if data_tmp:
        data.append(["ID:" + str(cnt), "Turn:" + str(int((len(data_tmp) - 1) / 2))] + data_tmp)
    return data


def raw2json():
    for i in range(2010, 2021):
        file_path = "./{}.txt".format(str(i))
        data = processing(file_path)
        data = [i for i in data if i[1] != 'Turn:0']
        print(len(data))
        with open("./{}.json".format(str(i)), 'w', encoding='utf-8') as file:
            file.write(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raw2json()
    import os
    import json
    import random

    data = []
    for a, b, file in os.walk('./'):
        for f in file:
            if f.endswith("json") and not f.startswith("train"):  # 轮数大于5的就不要了
                print(f)
                data.extend([i for i in json.load(open(f, "r", encoding="utf8"))
                             if len(i) > 2 and "科室" in i[2] and int(i[1].split(":")[-1]) <= 5
                             and "||" not in i[3] and "|。|" not in i[3]])
    print(len(data))
    data_ = []  # 清洗病人医生轮流出错的脏数据
    for i in data:
        flag = False
        l = len(i)
        for j in range(3, l):
            if j % 2 == 1 and i[j][:2] == "医生":
                flag = True
                break
            if j % 2 == 0 and i[j][:2] == "病人":
                flag = True
                break
        if flag:
            break
        data_.append(i)
    print(len(data_))
    data = []  # 重塑科室、疾病、描述
    for i in data_:
        d = i[3][3:].split("|")
        dise = d[0]
        desc = d[1]
        ask = "病人：" + i[3].split("|病人：")[-1]
        data_tmp = i[:3]
        data_tmp.append("疾病：" + dise)
        data_tmp.append("描述：" + desc)
        if len(ask) < 4:
            break
        data_tmp.append(ask)
        data_tmp.extend(i[4:])
        data.append(data_tmp)
    print(len(data))

    random.shuffle(data)
    train_data = data[:100000]
    validate_data = data[100000:110000]
    test_data = data[110000:120000]
    train_data_dpt = [[i[5][:3] + i[2][3:] + "," + i[3][3:] + "," + i[5][3:]] + i[6:] for i in train_data]
    validate_data_dpt = [[i[5][:3] + i[2][3:] + "," + i[3][3:] + "," + i[5][3:]] + i[6:] for i in validate_data]
    test_data_dpt = [[i[5][:3] + i[2][3:] + "," + i[3][3:] + "," + i[5][3:]] + i[6:] for i in test_data]

    train_data_info = [[i[5][:3] + i[2][3:] + "," + i[3][3:] + "," + i[4][3:] + i[5][3:]] + i[6:] for i in train_data]
    validate_data_info = [[i[5][:3] + i[2][3:] + "," + i[3][3:] + "," + i[4][3:] + i[5][3:]] + i[6:] for i in
                          validate_data]
    test_data_info = [[i[5][:3] + i[2][3:] + "," + i[3][3:] + "," + i[4][3:] + i[5][3:]] + i[6:] for i in test_data]

    train_data_new = [[i[2][3:] + "," + i[3][3:] + "," + i[4][3:]] + i[5:] for i in train_data]
    validate_data_new = [[i[2][3:] + "," + i[3][3:] + "," + i[4][3:]] + i[5:] for i in validate_data]
    test_data_new = [[i[2][3:] + "," + i[3][3:] + "," + i[4][3:]] + i[5:] for i in test_data]
    with open("./data_json/train_data_dpt.json", 'w', encoding='utf-8') as file:
        file.write(json.dumps(train_data_dpt, indent=2, ensure_ascii=False))
    with open("./data_json/validate_data_dpt.json", 'w', encoding='utf-8') as file:
        file.write(json.dumps(validate_data_dpt, indent=2, ensure_ascii=False))
    with open("./data_json/test_data_dpt.json", 'w', encoding='utf-8') as file:
        file.write(json.dumps(test_data_dpt, indent=2, ensure_ascii=False))

    with open("./data_json/train_data_info.json", 'w', encoding='utf-8') as file:
        file.write(json.dumps(train_data_info, indent=2, ensure_ascii=False))
    with open("./data_json/validate_data_info.json", 'w', encoding='utf-8') as file:
        file.write(json.dumps(validate_data_info, indent=2, ensure_ascii=False))
    with open("./data_json/test_data_info.json", 'w', encoding='utf-8') as file:
        file.write(json.dumps(test_data_info, indent=2, ensure_ascii=False))

    with open("./data_json/train_data_new.json", 'w', encoding='utf-8') as file:
        file.write(json.dumps(train_data_new, indent=2, ensure_ascii=False))
    with open("./data_json/validate_data_new.json", 'w', encoding='utf-8') as file:
        file.write(json.dumps(validate_data_new, indent=2, ensure_ascii=False))
    with open("./data_json/test_data_new.json", 'w', encoding='utf-8') as file:
        file.write(json.dumps(test_data_new, indent=2, ensure_ascii=False))





