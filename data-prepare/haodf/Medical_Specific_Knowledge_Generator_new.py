from py2neo import Graph
# from zhon.hanzi import punctuation
import json
import re
from distance import HanMingANDEdit
import time

class Process(object):
    def __init__(self):
        """ Knowledge Graph Loading """
        self.g = Graph(
            "bolt://106.14.44.63:7687",
            user="neo4j",
            password="wang123"
        )

        """ Adding Key Corresponding Infomation"""
        self.kw = {"dise":["症","白内障","中毒","综合征","癌","瘤","肺炎","结核","膜炎","损伤","胃炎","囊肿","低血糖",
                           "高血糖","骨折","病","感冒","哮喘","梗死","龟头炎","宫颈炎","心衰竭","白血","甲亢","结石"],
                   "sym": ["不舒服","难受","疼","痒","痛","酸","红","青","烫","凉","冷","热","无力","吐",
                           "呕","饿","脓","昏","塞","烧","恶心","臭","色","屎","尿","大便","小便"],
                   "food": ["食","吃东西","忌口","果","菜","汤","点心"],
                   "drug": ["药","服用","剂","胶囊","素"],
                   "check": ["检查", "验", "体检", "B超", "ct", "CT", "X光", "拍片"]
                   }

        with open("key_body_symprelated.json", 'r', encoding='utf8') as f:
            self.key_symp_related = json.load(f)
        self.key_symp_class = list(self.key_symp_related.keys())

        with open("dpt_text2kg.json", 'r', encoding='utf8') as f:
            self.dpt_text2kg = json.load(f)

        with open("kg_disease.txt", 'r', encoding='utf8') as f:
            self.kg_dise = [i.strip() for i in f]

        with open("kg_symptom.txt", 'r', encoding='utf8') as f:
            self.kg_symp_all = [i.strip() for i in f]

        """ Similarity of Entity (Replacable) """
        self.dis_dise= HanMingANDEdit(self.kg_dise)
        self.dis_symp = HanMingANDEdit(self.kg_symp_all)
        self.dis_dise_symp = HanMingANDEdit(self.kg_dise + self.kg_symp_all)

    # def seen_feature_extraction(self, input_file,i,DR_iminus1,output_file):

    def seen_feature_extraction(self, data, turn_id=0):
        tmp = data[turn_id].split(",")
        Dpt_seen = tmp[0]
        Dise_Sym_seen = tmp[1]
        ## Initial Patient's Question
        if turn_id == 0:
            PQ_i = "病人：" + ",".join(tmp[2:]) + data[1][3:] + "," + "。".join([data[i][3:] for i in range(len(data)) if i%2==1 and i!=1])
            DR_iminus1 = ''
        else:
            PQ_i = "病人：" + data[2*turn_id-1][3:]
            DR_iminus1 = ""
        MedKInfo = {"flag": ["disease"], "dpt": [], "kg_dpt": [],"hdpt": [],
                    "dise": [],"kg_dise": [],"hdise": [],
                    "sym": [],"kg_sym": [], "hsym": [],
                    "check": [],"kg_check": [], "hcheck": [],
                    "drug": [], "kg_drug": [],"hdrug": [],
                    "food_pos": [],"kg_food_pos": [],"hfood_pos": [],
                    "food_neg": [],"kg_food_neg": [],"hfood_neg": []}
        return Dpt_seen, Dise_Sym_seen, PQ_i, DR_iminus1, MedKInfo

    def fetch_hidden(self,merge_score,rank_hop):
        score_value = [i[1] for i in merge_score]
        score_value = sorted(list(set(score_value)))
        kg_feature = [i[0] for i in merge_score if i[1] == score_value[0]]
        if rank_hop > 1:
            hidden_feature = [i[0] for i in merge_score if i[1] == score_value[rank_hop-1]]
        else:
            hidden_feature = None
        return kg_feature, hidden_feature

    def map_feature(self,feature,KG_Cor_F2F):
        try:
            f2 = [KG_Cor_F2F[f1] for f1 in feature]
            f2 = list(set(f2))
        except:
            f2 = list()
        return f2

    def check_flag(self,type,PQ_i,kg_type,possibile_setence_seen):
        flag = False
        feature_s,feature_e,kg_feature,hfeature = [],[],[],[]

        if "pos" in type or "neg" in type:
            rep = type[-3:len(type)]
            type = type.replace(rep,'')

        for key_word in self.kw[type]:
            if key_word in PQ_i:
                for sentence in possibile_setence_seen:
                    if key_word in sentence:
                        if sentence.startswith(key_word):
                            sen_kw = sentence[:len(key_word) + 3]
                        elif sentence.endswith(key_word):
                            sen_kw = sentence[-len(key_word)-3:]
                        else:
                            sen_kw_list = sentence.split(key_word)
                            sen_kw = sen_kw_list[0][-3:] + key_word +sen_kw_list[1][:3]
                        feature_s.append(sentence)
                        feature_e.append(sen_kw)
                        if (key_word in sentence)&(type != "food"):
                            flag = True
                            break
                        elif (key_word in sentence)&(type == "food"):
                            if (("不" or "注意" or "禁" or "忌") in sentence):
                                food_neg_flag = True
                            else:
                                food_pos_flag = True
                            break
                        break

        if len(kg_type) != 0:
            for e in feature_e:
                flag_dis = HanMingANDEdit(kg_type)
                hm_score = flag_dis.hanming_distance(e)
                if hm_score and hm_score[0][1] == 0:
                    pass
                else:
                    edit_score = flag_dis.edit_dist(e)  ## 编辑距离
                    merge_score = flag_dis.merge_symp(edit_score, hm_score)  ## 汉明+编辑距离 = Dist1*k1 + Dist2*k2 (k1=1/10,k2=-1)
                    if merge_score:
                        kg_feature_temp, _ = self.fetch_hidden(merge_score, rank_hop=1)
                        hfeature_temp = list(set(kg_type).difference(set(kg_feature_temp)))
                        kg_feature.extend(kg_feature)
                        hfeature.extend(hfeature_temp)
        return feature_s, feature_e, kg_feature, hfeature


    def search(self,Dpt_seen,Dise_Sym_seen,PQ_i,DR_iminus1,MedKInfo):
        """
        Input: Seen Features
               Dpt_seen: Department
               Dise_Sym_seen: Disease/Symptoms
               PQ_i: Patient's Question in i turn (PQ)
               DR_iminus1: Doctor's Response in i turn (PQ)
        Output: Seen + Unseen Features
                MedKInfo: result (Knowledge Tuple)
        """
        #######################################################################################################
        ## Find Dpt and Sym and Dise
        Dise_Sym_seen = "".join(re.findall('[\u4e00-\u9fa5]', Dise_Sym_seen))
        if ("不" in Dise_Sym_seen and ("孕" not in Dise_Sym_seen or "育" not in Dise_Sym_seen)) or "未知" in Dise_Sym_seen:
            Dise_Sym_seen = None
        # Dise_Sym_seen = None
        # Dise_Sym_seen = "脸一边大一边小 很明显"
        if Dise_Sym_seen == None:# 如果没有疾病，就去KG寻找科室下的所有症状，看看有没有症状在句子中,在句子中的症状、疾病作为信息
            MedKInfo["flag"] = ["sym"]
            t1=time.time()
            KG_Cor_Dpt = self.dpt_text2kg[Dpt_seen] #kg_dpt = self.dpt_text2kg[Dpt_seen]
            MedKInfo["dpt"].append(Dpt_seen)
            MedKInfo["kg_dpt"] = KG_Cor_Dpt
            KG_Cor_Dise_tmp = [] #kg_dise_tmp = []
            sql = ["MATCH (m:Disease)-[r:belongs_to]->(n:Department) where n.name = '{0}'\
                                                RETURN m.name, r.name, n.name LIMIT 20".format(i) for i in KG_Cor_Dpt]
            for q in sql:
                KG_Cor_Dise_tmp += [i["m.name"] for i in self.g.run(q).data()]
            sql = ["MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) where m.name = '{0}'\
                                                RETURN m.name, r.name, n.name LIMIT 20".format(i) for i in KG_Cor_Dise_tmp]
            KG_Cor_Sym2Dise = {}
            for q in sql:
                KG_Cor_Sym2Dise.update({i["n.name"]: i["m.name"] for i in self.g.run(q).data()})
            KG_Symp_list = list(KG_Cor_Sym2Dise.keys())
#             print("kg_search1: ", time.time()-t1)
            ## Seen Sympton in Sentence
            for key_word in self.kw["sym"]:
                if key_word in PQ_i:
                    PQ_i_Symp_numList = [m.start() for m in re.finditer(key_word,PQ_i)]
                    a,b,c,d,e = "？","！","。","，","："
                    PQ_i_tmp = PQ_i.replace(e,"@").replace(a,"@").replace(b,"@").replace(c,"@").replace(d,"@")
                    PQ_i_at_numList = [m.start() for m in re.finditer("@", PQ_i_tmp)]
                    PQ_i_at_numList.append(len(PQ_i_tmp))
                    for PQ_i_Symp_index in PQ_i_Symp_numList:
                        for att_index in range(len(PQ_i_at_numList)-1):
                            if PQ_i_Symp_index <= PQ_i_at_numList[att_index+1] and PQ_i_Symp_index >= PQ_i_at_numList[att_index]:
                                possibile_symp_seen=PQ_i[PQ_i_at_numList[att_index]+1:PQ_i_at_numList[att_index+1]]
                                for key_sympclass in self.key_symp_class:
                                    if key_sympclass in possibile_symp_seen:
                                        symp = possibile_symp_seen
                                        MedKInfo["sym"].append(symp)
    ## Correspond to Sympton in Sentence
                                        symp_tmp = symp
                                        if key_word == "不舒服":
                                            symp_tmp = symp.replace(key_word, "痛痒")
                                        if key_word == "屎":
                                            symp_tmp = symp.replace(key_word, "大便")
                                        if key_word == "尿":
                                            symp_tmp = symp.replace(key_word, "小便")
                                        key_symp_list = self.key_symp_related[key_sympclass]
                                        key_symp_possibile = [key_symp for key_symp in key_symp_list if key_symp in symp_tmp]
                                        if not key_symp_possibile:
                                            continue
                                        key_symp = key_symp_possibile[0]
                                        KG_Symp_real_list = [kg_symp_p for kg_symp_p in KG_Symp_list if key_symp in kg_symp_p]
                                        KG_Symp_remove_list = [element for element in KG_Symp_real_list for key_symp_no in key_symp_list if (key_symp_no != key_symp)&(key_symp_no not in self.key_symp_class)&(key_symp_no in element)]
                                        KG_Symp_real_list = list(set(KG_Symp_real_list).difference(set(KG_Symp_remove_list)))
                                        dis_symp = HanMingANDEdit(KG_Symp_real_list)
                                        hm_score = dis_symp.hanming_distance(symp_tmp)  ## 汉明距离
                                        if hm_score and hm_score[0][1] == 0:
                                            pass
                                        else:
                                            edit_score = dis_symp.edit_dist(symp_tmp)  ## 编辑距离
                                            merge_score = dis_symp.merge_symp(edit_score, hm_score)  ## 汉明+编辑距离 = Dist1*k1 + Dist2*k2 (k1=1/10,k2=-1)
                                            if merge_score:
                                                kg_symp, hidden_symp = self.fetch_hidden(merge_score,rank_hop=2)
                                                MedKInfo["kg_sym"] = kg_symp
                                                MedKInfo["hsym"] = hidden_symp
                                                MedKInfo["kg_dise"] = self.map_feature(kg_symp, KG_Cor_Sym2Dise)
                                                MedKInfo["hdise"] = self.map_feature(hidden_symp, KG_Cor_Sym2Dise)
                                        break
            debug_1 = 000
        else:
            KG_Cor_Dpt = self.dpt_text2kg[Dpt_seen]
            MedKInfo["dpt"].append(Dpt_seen)
            MedKInfo["kg_dpt"] = KG_Cor_Dpt
            #### Check whether Dise_Symp_seen is disease or symptoms
            true_dise = [i for i in self.kw["dise"] if i in Dise_Sym_seen]
            true_symp = [i for i in self.kw["sym"] if i in Dise_Sym_seen]
            if len(true_dise) != 0 and len(true_symp) == 0:
                check_Dise_Symp = 'dise'
            elif len(true_dise) == 0 and len(true_symp) != 0:
                check_Dise_Symp = 'sym'
            else:
                check_Dise_Symp = 'unsure'
                hm_score_dise_symp = self.dis_dise_symp.hanming_distance(Dise_Sym_seen)  ## 汉明距离
                if hm_score_dise_symp and hm_score_dise_symp[0][1] == 0:
                    pass
                else:
                    edit_score_dise_symp = self.dis_dise_symp.edit_dist(Dise_Sym_seen)  ## 编辑距离
                    merge_score_dise_symp = self.dis_dise_symp.merge_symp(edit_score_dise_symp, hm_score_dise_symp)
                    if merge_score_dise_symp:
                        kg_best_match, _ = self.fetch_hidden(merge_score_dise_symp,rank_hop=1)
                        dise_num = [i for i in kg_best_match if i in self.kg_dise]
                        sym_num = [i for i in kg_best_match if i in self.kg_symp_all]
                        if dise_num >= sym_num:
                            check_Dise_Symp = 'dise'
                        else:
                            check_Dise_Symp = 'sym'
            if check_Dise_Symp == 'unsure':
                MedKInfo["dise"] = Dise_Sym_seen
                return MedKInfo
            elif check_Dise_Symp == 'dise':
                MedKInfo["dise"].append(Dise_Sym_seen)
                hm_score_dise = self.dis_dise.hanming_distance(Dise_Sym_seen)  ## 汉明距离
                if hm_score_dise and hm_score_dise[0][1] == 0:
                    pass
                else:
                    edit_score_dise = self.dis_dise.edit_dist(Dise_Sym_seen)  ## 编辑距离
                    merge_score_dise = self.dis_dise.merge_dise(edit_score_dise, hm_score_dise)  ## 汉明+编辑距离 = Dist1*k1 + Dist2*k2 (k1=1,k2=1)
                    if KG_Cor_Dpt:  ## 检查距离矩阵里的疾病是否在知识图谱的科室下
                        t1 = time.time()
                        kg_tmp = []
                        sql = ["MATCH (m:Disease)-[r:belongs_to]->(n:Department) where n.name = '{0}'\
                                            RETURN m.name, r.name, n.name".format(i) for i in KG_Cor_Dpt]
                        for q in sql:
                            kg_tmp += [i["m.name"] for i in self.g.run(q).data()]
#                         print("kg_search3: ", time.time()-t1)
                        merge_score_dise = [i for i in merge_score_dise if i[0] in kg_tmp]
                    if merge_score_dise:
                        kg_dise, _ = self.fetch_hidden(merge_score_dise, rank_hop=1)
                        MedKInfo["kg_dise"] = kg_dise
                        kg_sym = []
                        hsym = []
                        for kg_dise_single in kg_dise:
                            t1=time.time()
                            sql = "MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) where m.name = '{0}'\
                                RETURN m.name, r.name, n.name".format(kg_dise_single)
                            kg_sym_whole = [i["n.name"] for i in self.g.run(sql).data()]
#                             print("kg_search: ", time.time()-t1)
                            ## Seen Sympton in Sentence
                            for key_word in self.kw["sym"]:
                                if key_word in PQ_i:
                                    PQ_i_Symp_numList = [m.start() for m in re.finditer(key_word, PQ_i)]
                                    a, b, c, d, e = "？", "！", "。", "，", "："
                                    PQ_i_tmp = PQ_i.replace(e, "@").replace(a, "@").replace(b, "@").replace(c,"@").replace(d, "@")
                                    PQ_i_at_numList = [m.start() for m in re.finditer("@", PQ_i_tmp)]
                                    PQ_i_at_numList.append(len(PQ_i_tmp))
                                    for PQ_i_Symp_index in PQ_i_Symp_numList:
                                        for att_index in range(len(PQ_i_at_numList) - 1):
                                            if PQ_i_Symp_index <= PQ_i_at_numList[att_index + 1] and PQ_i_Symp_index >= PQ_i_at_numList[att_index]:
                                                possibile_symp_seen = PQ_i[PQ_i_at_numList[att_index] + 1:PQ_i_at_numList[att_index + 1]]
                                                for key_sympclass in self.key_symp_class:
                                                    if key_sympclass in possibile_symp_seen:
                                                        symp = possibile_symp_seen
                                                        MedKInfo["sym"].append(symp)
                                                        ## Correspond to Sympton in Sentence
                                                        symp_tmp = symp
                                                        if key_word == "不舒服":
                                                            symp_tmp = symp.replace(key_word, "痛痒")
                                                        if key_word == "屎":
                                                            symp_tmp = symp.replace(key_word, "大便")
                                                        if key_word == "尿":
                                                            symp_tmp = symp.replace(key_word, "小便")
                                                        if not symp_tmp:
                                                            continue
                                                        key_symp_list = self.key_symp_related[key_sympclass]
                                                        key_symp_possibile = [key_symp for key_symp in key_symp_list if key_symp in symp_tmp]
                                                        if not key_symp_possibile:
                                                            continue
                                                        key_symp = key_symp_possibile[0]
                                                        KG_Symp_real_list = [kg_symp_p for kg_symp_p in kg_sym_whole if key_symp in kg_symp_p]
                                                        KG_Symp_remove_list = [element for element in KG_Symp_real_list for key_symp_no in key_symp_list if (key_symp_no != key_symp) & (key_symp_no not in self.key_symp_class) & (key_symp_no in element)]
                                                        KG_Symp_real_list = list(set(KG_Symp_real_list).difference(set(KG_Symp_remove_list)))
                                                        dis_symp = HanMingANDEdit(KG_Symp_real_list)
                                                        hm_score = dis_symp.hanming_distance(symp_tmp)  ## 汉明距离
                                                        if hm_score and hm_score[0][1] == 0:
                                                            pass
                                                        else:
                                                            edit_score = dis_symp.edit_dist(symp_tmp)  ## 编辑距离
                                                            merge_score = dis_symp.merge_symp(edit_score,hm_score)  ## 汉明+编辑距离 = Dist1*k1 + Dist2*k2 (k1=1/10,k2=-1)
                                                            if merge_score:
                                                                kg_symp,_ = self.fetch_hidden(merge_score,rank_hop=1)
                                                                hsymp = list(set(kg_sym_whole).difference(set(kg_symp)))
                                                                kg_sym = kg_sym + kg_symp
                                                                hsym = hsym + hsymp
                                                        break
                        MedKInfo["kg_sym"] = kg_sym
                        MedKInfo["hsym"] = hsym
                debug_2 = 000
            elif check_Dise_Symp == 'sym':
                MedKInfo["sym"].append(Dise_Sym_seen)
                hm_score_sym = self.dis_symp.hanming_distance(Dise_Sym_seen)  ## 汉明距离
                if hm_score_sym and hm_score_sym[0][1] == 0:
                    pass
                else:
                    edit_score_sym = self.dis_symp.edit_dist(Dise_Sym_seen)  ## 编辑距离
                    merge_score_sym = self.dis_symp.merge_symp(edit_score_sym,hm_score_sym)  ## 汉明+编辑距离 = Dist1*k1 + Dist2*k2 (k1=1,k2=1)
                    if merge_score_sym:
                        kg_sym, hidden_sym = self.fetch_hidden(merge_score_sym, rank_hop=2)
                        MedKInfo["kg_sym"] = kg_sym
                        MedKInfo["hsym"] = hidden_sym ##要加入新的限制 器官名词
                        KG_Cor_Dise_tmp = []  # kg_dise_tmp = []
                        t1=time.time()
                        sql = ["MATCH (m:Disease)-[r:belongs_to]->(n:Department) where n.name = '{0}'\
                                                                        RETURN m.name, r.name, n.name LIMIT 20".format(i) for i in KG_Cor_Dpt]
                        for q in sql:
                            KG_Cor_Dise_tmp += [i["m.name"] for i in self.g.run(q).data()]
                        sql = ["MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) where m.name = '{0}'\
                                                                        RETURN m.name, r.name, n.name LIMIT 20".format(i) for i in KG_Cor_Dise_tmp]
                        KG_Cor_Sym2Dise = {}
                        for q in sql:
                            KG_Cor_Sym2Dise.update({i["n.name"]: i["m.name"] for i in self.g.run(q).data()})
#                         print("kg_search2: ", time.time()-t1)
                        kg_dise = []
                        hdise = []
                        kg_dise = kg_dise + self.map_feature(kg_sym, KG_Cor_Sym2Dise)

                        hdise = hdise + self.map_feature(hidden_sym, KG_Cor_Sym2Dise)
                        MedKInfo["kg_dise"] = kg_dise
                        MedKInfo["hdise"] = hdise
                debug_3 = 000
        #######################################################################################################
        ## Find Drug
        ## Split the sentence
        a, b, c, d, e = "？", "！", "。", "，", "："
        PQ_i_tmp = PQ_i.replace(e, "@").replace(a, "@").replace(b, "@").replace(c, "@").replace(d, "@")
        PQ_i_at_numList = [m.start() for m in re.finditer("@", PQ_i_tmp)]
        PQ_i_at_numList.append(len(PQ_i_tmp))
        possibile_setence_seen = []
        for att_index in range(len(PQ_i_at_numList) - 1):
            possibile_setence_seen.append(PQ_i[PQ_i_at_numList[att_index] + 1:PQ_i_at_numList[att_index + 1]])
        ## Find symptom ->drugs
        sql = ["MATCH (m:Disease)-[r:common_drug]->(n:Drug) where m.name = '{0}'\
                            RETURN m.name, r.name, n.name".format(i) for i in MedKInfo["kg_dise"]]
        kg_drug = []
        for q in sql:
            kg_drug += [i["n.name"] for i in self.g.run(q).data()]
        ## Find symptom ->check
        sql = ["MATCH (m:Disease)-[r:need_check]->(n:Check) where m.name = '{0}'\
                              RETURN m.name, r.name, n.name".format(i) for i in MedKInfo["kg_dise"]]
        kg_check = []
        for q in sql:
            kg_check += [i["n.name"] for i in self.g.run(q).data()]
        ## Find symptom ->food_positive
        sql = ["MATCH (m:Disease)-[r:do_eat]->(n:Food) where m.name = '{0}'\
                              RETURN m.name, r.name, n.name".format(i) for i in MedKInfo["kg_dise"]]
        kg_food_pos = []
        for q in sql:
            kg_food_pos += [i["n.name"] for i in self.g.run(q).data()]

        sql = ["MATCH (m:Disease)-[r:no_eat]->(n:Food) where m.name = '{0}'\
                            RETURN m.name, r.name, n.name".format(i) for i in MedKInfo["kg_dise"]]
        kg_food_neg = []
        for q in sql:
            kg_food_neg += [i["n.name"] for i in self.g.run(q).data()]

        drug_sentence, drug_feature,kg_drug_feature,hdrug_feature= self.check_flag("drug", PQ_i, kg_drug,possibile_setence_seen)
        check_sentence, check_feature,kg_check_feature,hcheck_feature = self.check_flag("check", PQ_i, kg_check,possibile_setence_seen)
        food_pos_sentence, food_pos_feature,kg_food_pos_feature,hfood_pos_feature = self.check_flag("foodpos", PQ_i, kg_food_pos,possibile_setence_seen)
        food_neg_sentence, food_neg_feature,kg_food_neg_feature,hfood_neg_feature = self.check_flag("foodneg", PQ_i, kg_food_neg,possibile_setence_seen)

        MedKInfo["check"] = check_feature
        MedKInfo["drug"] = drug_feature
        MedKInfo["food_pos"] = food_pos_feature
        MedKInfo["food_neg"] = food_neg_feature
        MedKInfo["kg_check"] = kg_check_feature
        MedKInfo["kg_drug"] = kg_drug_feature
        MedKInfo["kg_food_pos"] = kg_food_pos_feature
        MedKInfo["kg_food_neg"] = kg_food_neg_feature
        MedKInfo["hcheck"] = hcheck_feature
        MedKInfo["hdrug"] = hdrug_feature
        MedKInfo["hfood_pos"] = hfood_pos_feature
        MedKInfo["hfood_neg"] = hfood_neg_feature
        MedKInfo["check_sentence"] = check_sentence
        MedKInfo["drug_sentence"] = drug_sentence
        MedKInfo["food_pos_sentence"] = food_pos_sentence
        MedKInfo["food_neg_sentence"] = food_neg_sentence
        
    def multi(self, data, MedKInfo):
        MedKInfo_new = []
        sym = MedKInfo.get("sym")
        check = MedKInfo.get("check")
        drug = MedKInfo.get("drug")
        food_pos = MedKInfo.get("food_pos")
        food_neg = MedKInfo.get("food_neg")
        candidate = {"sym":sym, "check": check, "drug":drug, "food_pos":food_pos, "food_neg":food_neg}
        for i in range(int((len(data)-1)/2)):
            # food_pos_flag, food_neg_flag, drug, check = False, False, False, False
            if i == 0:
                d = data[0] + "," + data[1][3:]
            else:
                d = data[2*i + 1]
            d = d.replace(",", "@")
            d = d.replace("，", "@")
            d = d.replace("。", "@")
            d = d.replace("!", "@")
            d = d.replace(".", "@")
            d = d.replace("呢", "？")
            d = d.replace("吗", "？")
            d = d.replace("还是", "？")
            d = d.replace("可不可以", "？")
            d_list = [i for i in d.split("@") if i]
            d_list[-1] = d_list[-1] + "?"
            MedKInfo_tmp = {"sym":[],"hsym":[],
                "check":[],"hcheck":[],"drug":[],"hdrug":[],
                "food_pos":[],"hfood_pos":[],"food_neg":[],"hfood_neg":[]
            }
            for key, cdc in candidate.items():
                if not cdc:
                    continue
                for c in cdc:
                    for d_ in d_list:
                        if c in d_:
                            MedKInfo_tmp[key].append(c)
                            if "?" in d_ or "？" in d_:
                                MedKInfo_tmp["h"+key] = MedKInfo["h"+key]
            MedKInfo_new.append(MedKInfo_tmp)
        return MedKInfo_new    

    def main(self, input_file, output_file):
        with open(input_file, 'r', encoding='utf8') as f:
            load_dict = json.load(f)
        load_dict_new_lk = []
        cnt_sym, cnt_check, cnt_food_pos, cnt_food_pos, cnt_drug = 0, 0, 0, 0, 0
        for i, data in enumerate(load_dict):
            if i>=4000:
                continue
            if i % 10 == 0:
                print(i)
            Dpt_seen, Dise_Sym_seen, PQ_i, DR_iminus1, MedKInfo = self.seen_feature_extraction(data, turn_id=0)
            try:
                self.search(Dpt_seen, Dise_Sym_seen, PQ_i, DR_iminus1, MedKInfo)
                MedKInfo = {key: list(set(MedKInfo[key])) for key in MedKInfo}
                MedKInfo_new = self.multi(data, MedKInfo)
                MedKInfo_new = [{key: list(set(MI[key])) for key in MI} for MI in MedKInfo_new]
            except:
                print(i)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                MedKInfo = {}
            load_dict_new_lk.append([MedKInfo] + [MedKInfo_new] + data)
#             if i % 2000 == 0 and i != 0:
#                 time.sleep(2)
#                 with open(output_file+str(i)+".json", 'w', encoding='utf-8') as file:
#                     file.write(json.dumps(load_dict_new_lk[i-2000:i], indent=2, ensure_ascii=False))

        with open(output_file+"_0-4000.json", 'w', encoding='utf-8') as file:
            file.write(json.dumps(load_dict_new_lk, indent=2, ensure_ascii=False))


if __name__=="__main__":
    input_file = "test_data_new.json"
    # input_file = "demo.json"
    output_file = "./result_multi/load_dict_new_lk_test"
    p = Process()
    p.main(input_file, output_file)