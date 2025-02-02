
import json
# import cv2
from pprint import pprint

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
class Mistral:
    def __init__(self, device="cuda:0"):
        model_name = "../checkpoints/MiniGPT4-Video/Mistral-7B-Instruct-v0.2"
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
    
    def generate(self, messages, max_new_tokens=1000):

        # encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        out = self.tokenizer(
            [messages],
            max_length=max_new_tokens,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(**out, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        decoded = self.tokenizer.batch_decode(generated_ids)
        
        return decoded[0]

    def post_process(self, res,anchor):
        
        anchor_index = res.rfind(anchor)
        pred_string = ""  # if decoding failed, return ""
        if anchor_index >= 0:
            pred_string = res[anchor_index + len(anchor):].strip()

        if pred_string.endswith("</s>"):
            pred_string = pred_string[:-4]

        return pred_string

    
    def extral_answer(self, pred, qa, max_new_tokens=200):

        messages = (
            "<s>Below is an analysis of multiple topic (A, B, C), which includes answers and extracts them</s>"
            f"[INST]Question:{qa} \n Analysis:{pred} [/INST] Answer:("
            )
        output=self.generate(messages, max_new_tokens=max_new_tokens)
        output = self.post_process(output,anchor = "[/INST] Answer:(")
        
        return output
    
    def extral_answer_dis(self, pred, qa, max_new_tokens=2000):
            
        messages = (
            "<s>Based on the given discrimination  questions and a provided preliminary answer, extract whether the answer is 'yes' or 'no'. Respond directly without giving too much explanation.</s>"
            f"[INST]Question:{qa}\nPreliminary answer:{pred}\n[/INST] "
            )
        output=self.generate(messages, max_new_tokens=max_new_tokens)
        # print("output:",output)
        output = self.post_process(output,anchor = "[/INST]")
        # print("extral_answer_dis:",output)
        return output
    
class LLM:
    llm=None

class DisEvalKit:
    def __init__(self,output_path=None,need_llm_extral=False,device="cuda:0"):
        if need_llm_extral:
            if LLM.llm is None:
                LLM.llm=Mistral(device=device)
            self.llm=LLM.llm
        else:
            self.llm=None
            
        self.pred_list = []
        self.label_list = []
        self.class_list = []
        self.error_list = []

        self.results_dict=None
        self.pair_id_map={}

        if output_path:

            with open(output_path,'r') as f:
                data_json = json.load(f)
                for item in tqdm(data_json):
                    if item.get('predict') is None:
                        self.error_list.append(item['video_id'])
                        print("No predict found in item:",item['video_id'])
                        continue
                    item['predict_type']=self.recorder(item['predict'],item['dis_answer'],class_id=item['motion_id'],pair_id=item['id'],qa=item['dis_question'])
            
            # save result
            with open(output_path,'w') as f:
                json.dump(data_json,f)
        


    def print_result(self,print_result=True):

        if self.results_dict is None:

            pos = 1
            neg = 0
            yes_ratio = self.pred_list.count(1) / len(self.pred_list)

            # -1
            nan_ratio = self.pred_list.count(-1) / len(self.pred_list)
            # unknown_ratio = pred_list.count(2) / len(pred_list)

            TP, TN, FP, FN = 0, 0, 0, 0
            for pred, label in zip(self.pred_list, self.label_list):
                
                if pred == pos and label == pos:
                    TP += 1
                elif pred == pos and label == neg:
                    FP += 1
                elif pred == neg and label == neg:
                    TN += 1
                elif pred == neg and label == pos:
                    FN += 1

            precision = float(TP) / float(TP + FP)
            recall = float(TP) / float(TP + FN)
            no_acc = TN / (TN + FP)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2*precision*recall / (precision + recall)
            acc = (TP + TN) / (TP + TN + FP + FN)

            pct_diff=(self.pred_list.count(1)-self.label_list.count(1))/len(self.label_list)
            fp_ratio=FP/(FP+TN)

            self.results_dict={
                "TP":TP,
                "TN":TN,
                "FP":FP,
                "FN":FN,
                "Accuracy":acc,
                "Precision":precision,
                "Recall":recall,  # YesAcc
                "F1":f1,
                "YesRation":yes_ratio,
                "NoAcc":no_acc,
                "PctDiff":pct_diff,
                "FPRatio":fp_ratio,
                "nanRatio":nan_ratio
            }

            # calculate the acc for each class
            if len(self.class_list) > 0:
                class_dict = {}
                for i, class_id in enumerate(self.class_list):
                    if class_id not in class_dict:
                        class_dict[class_id] = {"total":0,"right":0}
                    class_dict[class_id]["total"] += 1
                    if self.pred_list[i] == self.label_list[i]:
                        class_dict[class_id]["right"] += 1
                for class_id in class_dict.keys():
                    class_dict[class_id]["acc"] = class_dict[class_id]["right"] / class_dict[class_id]["total"]
                self.results_dict["class_acc"] = class_dict

            if len(self.pair_id_map) > 0:
                total_pair = 0
                right_pair = 0
                for pair_id in self.pair_id_map.keys():
                    tmp_list=self.pair_id_map[pair_id]

                    have_nan=False
                    for i in range(len(tmp_list)):
                        if self.pred_list[tmp_list[i]] == -1:
                            have_nan=True
                            break
                    if have_nan:
                        continue
                    all_right=True
                    for i in range(len(tmp_list)):
                        if self.pred_list[tmp_list[i]] != self.label_list[tmp_list[i]]:
                            all_right=False
                            break
                    if all_right:
                        right_pair += 1
                    
                    total_pair += 1
                pair_acc = right_pair / total_pair
                self.results_dict["pair_acc"] = pair_acc

        
        if print_result:

    
            print('Accuracy: {}'.format(self.results_dict['Accuracy']))
            print('Recall: {}'.format(self.results_dict['Recall']))  # YesAcc

            print('No acc: {}'.format(self.results_dict['NoAcc']))
            print('Pct diff: {}'.format(self.results_dict['PctDiff']))
            print('FP ratio: {}'.format(self.results_dict['FPRatio']))
            print('nan ratio: {}'.format(self.results_dict['nanRatio']))

            if "pair_acc" in self.results_dict:
                print("Pair acc:{}".format(self.results_dict["pair_acc"]))

            print("--"*30)
 
        return self.results_dict

    def recorder(self,pred, label,class_id=None,pair_id=None,instruction=None):
        if label == "yes":
            self.label_list.append(1)
        elif label == "no":
            self.label_list.append(0)
        else:
            # self.label_list.append(-1)
            raise ValueError("Label not recognized")    
        
        NEG_WORDS = ["no","not"]
        POS_WORDS = ["yes"]

        pred=pred.lower()
        pred = pred.replace('.', '')
        pred = pred.replace(',', '')
        pred = pred.replace('\"', '')
        pred = pred.replace('\'', '')
        words = pred.split(' ')

        if class_id:
            self.class_list.append(int(class_id))
        if pair_id:
            if pair_id not in self.pair_id_map:
                self.pair_id_map[pair_id]=[]
            self.pair_id_map[pair_id].append(len(self.pred_list))

        if any(word in POS_WORDS for word in words):
            self.pred_list.append(1)

        elif any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            self.pred_list.append(0)
        
        else:
            if self.llm and instruction:
                pred=self.llm.extral_answer_dis(pred,qa=instruction)
                pred=pred.lower()
                pred = pred.replace('.', '')
                pred = pred.replace(',', '')
                pred = pred.replace('\"', '')
                pred = pred.replace('\'', '')
                words = pred.split(' ')
                if any(word in POS_WORDS for word in words):
                    self.pred_list.append(1)

                elif any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
                    self.pred_list.append(0)
                else:
                    self.pred_list.append(-1)
            else:
                self.pred_list.append(-1)
        

        
        return self.pred_list[-1]
    
    def save_result(self,save_path):
        # data = {'pred_list':self.pred_list,'label_list':self.label_list,'result_list':self.print_result(print_result=False),'error_list':self.error_list,'class_list':self.class_list}
        data = {'result_list':self.print_result(print_result=False)}
        with open(save_path,'w') as f:
            json.dump(data,f)
        print("Result saved to {}".format(save_path))

class ClsEvalKit:
    def __init__(self,result_path=None,output_path=None,need_llm_extral=False,device="cuda:0"):
        if need_llm_extral:
            if LLM.llm is None:
                LLM.llm=Mistral(device=device)
            self.llm=LLM.llm

        self.need_llm_extral=need_llm_extral
        self.ans_id_map={"A":0,"B":1,"C":2}

        if result_path:
            with open(result_path,'r') as f:
                data_json = json.load(f)
                self.pred_list = data_json['pred_list']
                self.label_list = data_json['label_list']
                self.classid_list = data_json['classid_list']
        elif output_path:
            self.pred_list = []
            self.label_list = []
            self.classid_list = []
            with open(output_path,'r') as f:
                data_json = json.load(f)
                for item in data_json:
                    item['predict_type']=self.recorder(item['predict'],item['answer'],instruction=None,class_id=item['motion_id'])
            
            # save result
            with open(output_path,'w') as f:
                json.dump(data_json,f)
        
        else:
            self.pred_list = []
            self.label_list = []
            self.classid_list = []
        self.results_dict=None
        
        

    def print_result(self,print_result=True):

        # -1
        if self.results_dict==None:
            nan_ratio = self.pred_list.count(-1) / len(self.pred_list)
            right_num = 0

            for pred, label in zip(self.pred_list, self.label_list):
                if pred == label:
                    right_num += 1

            acc = right_num / (len(self.pred_list) - self.pred_list.count(-1))
            acc = round(acc, 3)


            self.results_dict={
            "accuracy":acc,
            "nan_ratio":nan_ratio,
            "nan_num":self.pred_list.count(-1)
            }
            
            assert len(self.classid_list)>0

            classid_dict={}
            for i,class_id in enumerate(self.classid_list):
                if class_id not in classid_dict:
                    classid_dict[class_id]={"options_total":{0:0, 1:0, 2:0},"options_right":{0:0, 1:0, 2:0},"options_pred":{0:0, 1:0, 2:0}}
                if self.pred_list[i]==-1:
                    continue
                classid_dict[class_id]["options_total"][self.label_list[i]]+=1
                classid_dict[class_id]["options_pred"][self.pred_list[i]]+=1
                if self.pred_list[i]==self.label_list[i]:
                    classid_dict[class_id]["options_right"][self.label_list[i]]+=1
                    
            for class_id in classid_dict:
                # acc
                classid_dict[class_id]["accuracy"]=sum(classid_dict[class_id]["options_right"].values())/sum(classid_dict[class_id]["options_total"].values())
                classid_dict[class_id]["accuracy"]=round(classid_dict[class_id]["accuracy"], 3)
                # acc per option
                for option in classid_dict[class_id]["options_total"]:

                    classid_dict[class_id][f"option_{option}_accuracy"]=classid_dict[class_id]["options_right"][option]/classid_dict[class_id]["options_total"][option]
            
            # self.results_dict["classid_accuracy"]=classid_dict

            # acc per option
            options_dict={
                'A':self.compute_options_metrics(classid_dict,0),
                'B':self.compute_options_metrics(classid_dict,1),
                'C':self.compute_options_metrics(classid_dict,2)
            }

            # self.results_dict["options_dict"]=options_dict
            
            self.results_dict["macro_precision"]=(options_dict['A']['precision']+options_dict['B']['precision']+options_dict['C']['precision'])/3
            self.results_dict["macro_recall"]=(options_dict['A']['recall']+options_dict['B']['recall']+options_dict['C']['recall'])/3
            self.results_dict["macro_f1"]=(options_dict['A']['f1']+options_dict['B']['f1']+options_dict['C']['f1'])/3

        if print_result:
            print('Accuracy: {}'.format(self.results_dict['accuracy']))
            print('macro_precision: {}'.format(self.results_dict['macro_precision']))
            print('macro_recall: {}'.format(self.results_dict['macro_recall']))
            print('macro_f1: {}'.format(self.results_dict['macro_f1']))
            print('nan ratio: {}'.format(self.results_dict['nan_ratio']))
            print('nan num: {}'.format(self.results_dict['nan_num']))
            print("--"*30)


        return self.results_dict

    def compute_options_metrics(self, classid_dict,c_id):
        options_dict={}
        options_dict['rights'] = sum([classid_dict[class_id]["options_right"][c_id] for class_id in classid_dict]) 
        options_dict['totalgt']=sum([classid_dict[class_id]["options_total"][c_id] for class_id in classid_dict])
        options_dict['totalpred']=sum([classid_dict[class_id]["options_pred"][c_id] for class_id in classid_dict])
        if options_dict['totalgt'] == 0:
            options_dict['recall'] = 0
        else:
            options_dict['recall'] = options_dict['rights'] / options_dict['totalgt']
            
        if options_dict['totalpred'] == 0:
            options_dict['precision'] = 0
        else:
            options_dict['precision']=options_dict['rights'] / options_dict['totalpred']

        if options_dict['precision'] + options_dict['recall'] == 0:
            options_dict['f1'] = 0
        else:
            options_dict['f1'] = 2 * options_dict['precision'] * options_dict['recall'] / (options_dict['precision'] + options_dict['recall'])
        
        return options_dict

    def recorder(self,pred, label,instruction=None,class_id=None):

        if self.need_llm_extral:
            pred=self.llm.extral_answer(pred,instruction)
        
        if class_id:
            self.classid_list.append(int(class_id))

        self.label_list.append(self.ans_id_map[label])
 
        pred=pred.upper()
        pred =pred.strip()[0]
        if pred in self.ans_id_map:
            self.pred_list.append(self.ans_id_map[pred])
        else:
            self.pred_list.append(-1)
        
        return self.pred_list[-1]
    
    def save_result(self,save_path):
        # data = {'pred_list':self.pred_list,'label_list':self.label_list,'result_list':self.print_result(print_result=False),'classid_list':self.classid_list}
        data = {'result_list':self.print_result(print_result=False)}
        with open(save_path,'w') as f:
            json.dump(data,f)
        print("Result saved to {}".format(save_path))
