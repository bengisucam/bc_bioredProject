

import csv
import os
import re
import math
import pandas as pd
from nltk.tokenize import sent_tokenize

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


class Relation:
    def __init__(self, e1_id, e2_id, type, is_novel, is_relation, candidate_sentences):
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.type = type
        self.is_novel = is_novel
        self.is_relation = is_relation
        self.candidate_sentences = candidate_sentences


class Document:
    def __init__(self, id, text, relation_pairs):
        self.doc_id = id
        self.text = text
        self.relation_pairs = relation_pairs


class Entity:
    def __init__(self, locs, name, type, id):
        self.locs = locs
        self.name = name
        self.type = type
        self.id = id


class BioredData:
    def __init__(self, path, new_path=None):
        self.path = path
        self.new_path = new_path


# def getRelationDetail(line):
#
# def getEntityPair(line):


def getEntityDetails(line):
    # get the relation info:
    line_items = line.split("\t")
    entity = Entity(locs=[line_items[1], line_items[2]],
                    name=line_items[3],
                    type=line_items[4],
                    id=line_items[5])
    return entity


# if you compile the regex string first, it's even faster
RE_DIGIT = re.compile('\d')


def isDigit(string):
    return RE_DIGIT.search(string)


def addNewEntityAndUpdateNames(entity_dict, document_id, new_entity):
    entity_list = entity_dict[document_id]
    for e in entity_list:
        if e.id == new_entity.id and e.name != new_entity.name:
            if len(e.name) > len(new_entity.name):
                e.name = new_entity.name
            else:
                new_entity.name = e.name


def getAllPossibleRelationPairs(entity_list):
    possible_pairs = []
    for i in range(len(entity_list) - 1):
        first = entity_list[i]
        for j in range(i + 1, len(entity_list)):
            second = entity_list[j]
            possible_pairs.append([first, second])
    return possible_pairs


def createDocumentEntitiesAndRelations(in_pubtator_path):
    all_docs = {}
    all_entities = {}
    with open(os.path.join(os.getcwd(), in_pubtator_path), 'r', encoding='utf8') as pub_reader:
        line_count = 0
        for line in pub_reader:
            line_count += 1
            if line == '\n':
                line_count = 0
                continue
            line_items = line.split('\t')
            # combine the title and the abstract for each document id
            if line_count <= 2:
                doc_items = line.split('|')
                doc_id = doc_items[0]
                doc_text = doc_items[2]
                if doc_id not in all_docs:
                    # add the abstract of the document to the text
                    doc = Document(id=doc_id, text=doc_text, relation_pairs=[])
                    all_docs[doc_id] = doc
                else:
                    doc = all_docs[doc_id]
                    # add the title of the document to the text
                    doc.text = doc.text + " " + doc_text

            # get the entity information
            elif line_count > 2 and isDigit(line_items[1]):
                entity_id = line_items[-1].strip().split(',')
                doc_id = line_items[0]
                entity_loc = [int(line_items[1]), int(line_items[2])]
                entity_name = line_items[3]
                entity_type = line_items[4]
                for ei in entity_id:
                    if doc_id not in all_entities:
                        entity = Entity(id=ei, locs=entity_loc, name=entity_name, type=entity_type)
                        all_entities[doc_id] = [entity]
                    else:
                        new_entity = Entity(id=ei, locs=entity_loc, name=entity_name, type=entity_type)
                        all_entities[doc_id].append(new_entity)
                        if ei in [entity.id for entity in all_entities[doc_id]]:
                            addNewEntityAndUpdateNames(all_entities, doc_id, new_entity)
            # get the relation information
            elif line_count > 2 and not isDigit(line_items[1]):
                relation_type = line_items[1]
                entity1_id, entity2_id = line_items[2], line_items[3]
                is_relation_novel = True if line_items[-1].strip() == "Novel" else False
                # create True relation pairs
                relationObject = Relation(e1_id=entity1_id, e2_id=entity2_id, type=relation_type, is_relation=True,
                                          is_novel=is_relation_novel, candidate_sentences=None)
                all_docs[doc_id].relation_pairs.append(relationObject)

    # next, create False relation pairs
    for doc_id in all_entities:
        possible_pairs = getAllPossibleRelationPairs(all_entities[doc_id])
        true_relations = all_docs[doc_id].relation_pairs
        false_relation_list = []
        for pair in possible_pairs:
            entity1, entity2 = pair[0], pair[1]
            if len([rel for rel in true_relations if rel.e1_id == entity1.id and rel.e2_id == entity2.id]) != 0:
                continue
            elif len([rel for rel in true_relations if rel.e1_id == entity2.id and rel.e2_id == entity1.id]) != 0:
                continue
            else:
                false_relation = Relation(e1_id=entity1.id, e2_id=entity2.id, is_relation=False, type=None,
                                          is_novel=False,
                                          candidate_sentences=None)
                false_relation_list.append(false_relation)

        all_docs[doc_id].relation_pairs.extend(false_relation_list)

    return all_docs, all_entities


def removeEntitiesWithSameLocsButDifferentIDs(entity_list):
    new_list = []
    for i in range(len(entity_list) - 2):
        e1 = entity_list[i]
        e2 = entity_list[i + 1]
        if e1.locs == e2.locs:
            continue
        else:
            new_list.append(e1)
    return new_list


def normalizeEntityNames(entity_dict, doc_dict):
    for doc_id in doc_dict:
        entity_list = removeEntitiesWithSameLocsButDifferentIDs(entity_dict[doc_id])

        for entity in entity_list:
            start_loc, end_loc = entity.locs[0], entity.locs[1]
            old_name_len = end_loc - start_loc
            new_name_len = len(entity.name)
            diff = new_name_len - old_name_len
            if diff < 0:
                doc_dict[doc_id].text = doc_dict[doc_id].text[:start_loc] + entity.name + doc_dict[doc_id].text[
                                                                                          end_loc:]
                # update entities locations
                for e in entity_list:
                    # do not change the start index of the entity if it is the entity you update
                    if e == entity:
                        e.locs[1] = e.locs[1] + diff
                    elif e.locs[0] > entity.locs[1]:
                        e.locs[0], e.locs[1] = e.locs[0] + diff, e.locs[1] + diff
            elif diff > 0:
                continue
                #raise Exception("yeni entity ismi, eski entity isminden uzun olamaz")
            else:
                continue
    return doc_dict


def tokenize_sentences(context):
    sentences = sent_tokenize(context)
    return sentences


def create_sentence_combination(first_list, second_list):
    combined = []
    if first_list and second_list:
        for s1 in first_list:
            for s2 in second_list:
                combined_sentence = s1 + " " + s2
                combined.append(combined_sentence)
        return combined
    elif first_list:
        return first_list
    else:
        return second_list


def calculatePossibleNumberOfEntityPairs(entity_dictionary):
    total = 0
    for docid in entity_dictionary:
        unique_entity_set = set([e.id for e in entity_dictionary[docid]])
        total_doc_entities = len(unique_entity_set)
        total += math.comb(total_doc_entities, 2)
    return total


def get_total_num_of_documents(doc_path):
    doc_id_list= []
    with open(os.path.join(os.getcwd(), doc_path), 'r', encoding='utf8') as pub_reader:
        doc_count = 0
        is_docid_extracted = False
        for line in pub_reader:
            if line == '\n':
                doc_count += 1
                is_docid_extracted = False
            else:
                if not is_docid_extracted:
                    doc_items = line.split('|')
                    doc_id = doc_items[0]
                    doc_id_list.append(doc_id)
                    is_docid_extracted = True
    return doc_id_list


def have_common_item(list1, list2):
    for item in list1:
        if item in list2:
            return True
    return False

def replace_names_with_tags(folder_path):
    full_path = os.path.join(os.getcwd(), folder_path)
    all_files = os.listdir(full_path)
    for file in all_files:
        data = pd.read_csv(os.path.join(full_path, file))



if __name__ == '__main__':

    is_tagging_active = False

    in_data_dir = 'files'
    out_data_dir_in_sentence = 'kFoldFiles\\out_files_in_sentence'

    in_train_pubtator_file = in_data_dir + '\\Train.PubTator'
    in_dev_pubtator_file = in_data_dir + '\\Dev.PubTator'
    in_test_pubtator_file = in_data_dir + '\\Test.PubTator'

    if not os.path.exists(os.path.join(os.getcwd(), out_data_dir_in_sentence)):
        os.makedirs(out_data_dir_in_sentence)

    train_docs = get_total_num_of_documents(in_train_pubtator_file)
    dev_docs = get_total_num_of_documents(in_dev_pubtator_file)
    have_common_doc_id = have_common_item(train_docs, dev_docs)
    print(have_common_doc_id)

    doc_num = len(train_docs) + len(dev_docs)
    K = 10
    fold_doc_num = doc_num / K


    in_pubtator_file_list = [in_train_pubtator_file, in_dev_pubtator_file]
    out_file_list = ["train", "dev"]
    # out file's header and data information
    if is_tagging_active:
        headers = ["id", "docid", "entity1", "entity2", "isValid", "sentence", "preprocessedSentence"]
    else:
        headers = ["id", "docid", "entity1", "entity2", "isValid", "sentence"]
    fold_count = 1

    # create k-folds for only train and dev files, in sentence relations
    for i in range(0, len(in_pubtator_file_list)):
        out_file_prefix = out_file_list[i]
        # process train data
        doc_dict, entity_dict = createDocumentEntitiesAndRelations(in_pubtator_path=in_pubtator_file_list[i])
        normalized_doc_dict = normalizeEntityNames(entity_dict=entity_dict, doc_dict=doc_dict)

        # create out file
        data = []
        total_num_relations = 0
        total_num_true_relations = 0
        total_num_in_sentence_relations = 0
        total_num_true_in_sentence_relations = 0
        doc_count = 0

        for doc_id in normalized_doc_dict:
            doc_count += 1
            rel_id = 0
            entities: list[Entity] = entity_dict[doc_id]
            doc_as_sentences = tokenize_sentences(normalized_doc_dict[doc_id].text)
            normalized_doc_dict[doc_id].text = doc_as_sentences
            relation_pairs = normalized_doc_dict[doc_id].relation_pairs
            for relation in relation_pairs:
                if relation.is_relation:
                    total_num_true_relations += 1
                total_num_relations += 1
                e1 = [e for e in entities if e.id == relation.e1_id][0]
                e2 = [e for e in entities if e.id == relation.e2_id][0]
                sentences_with_both = [s for s in doc_as_sentences if e1.name in s and e2.name in s]
                # if there exists a intra-sentence pair, then no need to look for the inter-sentence pairs
                if sentences_with_both:
                    relation.candidate_sentences = sentences_with_both
                    # add the plain relation pair sentences into the out folder
                    # each sentence for same e1 and e2 will be counted as different relation pair sentences
                    for s in sentences_with_both:
                        data.append([rel_id, doc_id, e1.name, e2.name, relation.is_relation, s])
                        rel_id += 1
                        if relation.is_relation:
                            total_num_true_in_sentence_relations += 1
                        total_num_in_sentence_relations += 1

                    # todo:  find the sdp of the candidate sentences
            if doc_count % (doc_num / K) == 0:
                fold_path = out_data_dir_in_sentence + '\\' + out_file_prefix + 'Fold' + str(fold_count) + '.csv'
                with open(fold_path, 'w', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    writer.writerows(data)
                    f.close()
                fold_count += 1
                data.clear()
        print("#######{} Data Analysis########".format(out_file_list[i]))
        print("Total Number of  Relations: ", total_num_relations)
        print("Total Number of True Relations: ", total_num_true_relations)
        print("Total Number of  In Sentence Relations: ", total_num_in_sentence_relations)
        print("Total Number of True In Sentence Relations: ", total_num_true_in_sentence_relations)

    ######## TEST FILE ###############
    # create test file, in sentence relation pairs
    doc_dict, entity_dict = createDocumentEntitiesAndRelations(in_pubtator_path=in_test_pubtator_file)
    normalized_doc_dict = normalizeEntityNames(entity_dict=entity_dict, doc_dict=doc_dict)

    # create out file
    data = []
    total_num_relations_test = 0
    total_num_true_relations_test = 0
    total_num_in_sentence_relations_test = 0
    total_num_true_in_sentence_relations_test = 0
    doc_count = 0
    for doc_id in normalized_doc_dict:
        doc_count += 1
        rel_id = 0
        entities: list[Entity] = entity_dict[doc_id]
        doc_as_sentences = tokenize_sentences(normalized_doc_dict[doc_id].text)
        normalized_doc_dict[doc_id].text = doc_as_sentences
        relation_pairs = normalized_doc_dict[doc_id].relation_pairs
        for relation in relation_pairs:
            if relation.is_relation:
                total_num_true_relations_test += 1
            total_num_relations_test += 1
            e1 = [e for e in entities if e.id == relation.e1_id][0]
            e2 = [e for e in entities if e.id == relation.e2_id][0]
            sentences_with_both = [s for s in doc_as_sentences if e1.name in s and e2.name in s]
            # if there exists a intra-sentence pair, then no need to look for the inter-sentence pairs
            if sentences_with_both:
                relation.candidate_sentences = sentences_with_both
                # add the plain relation pair sentences into the out folder
                data.append([rel_id, doc_id, e1.name, e2.name, relation.is_relation, ''.join(relation.candidate_sentences)])
                rel_id += 1
                if relation.is_relation:
                    total_num_true_in_sentence_relations_test += 1
                total_num_in_sentence_relations_test += 1
                # todo:  find the sdp of the candidate sentences

        fold_path = out_data_dir_in_sentence + '\\' + 'test.csv'
        with open(fold_path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(headers)
            # write multiple rows
            writer.writerows(data)
            f.close()
        fold_count += 1
        data.clear()
    print("#######Test Data Analysis########")
    print("Total Number of  Relations: ", total_num_relations_test)
    print("Total Number of True Relations: ", total_num_true_relations_test)
    print("Total Number of  In Sentence Relations: ", total_num_in_sentence_relations_test)
    print("Total Number of True In Sentence Relations: ", total_num_true_in_sentence_relations_test)

    replace_names_with_tags(out_data_dir_in_sentence)