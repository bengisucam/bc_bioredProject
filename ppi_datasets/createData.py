import xml.etree.ElementTree as ET
import os
import pandas as pd


def create_all_entity_pairs(all_entity_list):
    pairs = []
    for i in range(len(all_entity_list) - 1):
        entity1 = all_entity_list[i]
        for j in range(i + 1, len(all_entity_list)):
            entity2 = all_entity_list[j]
            pairs.append([entity1, entity2])
    return pairs

def get_entity_offsets(entity_name_offset_dict, e_name):
    e_offsets = []
    offset_list = entity_name_offset_dict[e_name].split(",")
    if offset_list:
        for offset in offset_list:
            e_offsets.append(offset.split("-"))
    else:
        entity_name_offset_dict[e_name].split("-")
    return e_offsets

def tag_sentence_with_protein(passage, e1_offset_list, e2_offset_list, e1_name, e2_name):

    FIRST_E1 = True
    if int(e1_offset_list[0][0]) > int(e2_offset_list[0][0]):
        FIRST_E1 = False

    chars_to_update = 0
    if FIRST_E1:
        for e1_offset in e1_offset_list:
            e1_offset = [int(e1_offset[0]) + chars_to_update, int(e1_offset[1]) + chars_to_update]
            sentence_text = passage[:e1_offset[0]] + "<PROTEIN1>" + e1_name + "<PROTEIN1>" + passage[e1_offset[1]:]
            # update second entity's offset
            chars_to_update += 2 * len("<PROTEIN1>")

        for e2_offset in e2_offset_list:
            try:
                e2_offset = [int(e2_offset[0]) + chars_to_update, int(e2_offset[1]) + chars_to_update]
                sentence_text = sentence_text[:e2_offset[0]] + "<PROTEIN2>" + e2_name + "<PROTEIN2>" + sentence_text[e2_offset[1]:]
                chars_to_update += 2 * len("<PROTEIN2>")
            except IndexError:
                print()
    else:
        for e2_offset in e2_offset_list:
            e2_offset = [int(e2_offset[0]) + chars_to_update, int(e2_offset[1]) + chars_to_update]
            sentence_text = passage[:int(e2_offset[0])] + "<PROTEIN2>" + e2_name + "<PROTEIN2>" + passage[int(
                e2_offset[1]):]
            # update second entity's offset
            chars_to_update = 2 * len("<PROTEIN2>")
        for e1_offset in e1_offset_list:
            e1_offset = [int(e1_offset[0]) + chars_to_update, int(e1_offset[1]) + chars_to_update]
            sentence_text = sentence_text[:int(e1_offset[0])] + "<PROTEIN1>" + e1_name + "<PROTEIN1>" + sentence_text[int(
                e1_offset[1]):]
            chars_to_update = 2 * len("<PROTEIN1>")
    return sentence_text


def read_xml(folder_name, split):
    # Reading the data inside the xml
    file_name = folder_name + "-" + split + ".xml"
    file_path = os.path.join(os.getcwd(), folder_name + "\\" + file_name)
    tree = ET.parse(file_path)
    root = tree.getroot()
    all_docs = root.findall("document")

    # create dataframe for preprocessed sentences
    df = pd.DataFrame()

    for doc in all_docs:
        all_sentences = doc.findall("sentence")
        for sentence in all_sentences:
            all_entities = sentence.findall("entity")
            all_entities_id = [e.attrib["id"] for e in all_entities]
            all_entities_name = [e.attrib["text"] for e in all_entities]
            all_entities_id_names = dict(zip(all_entities_id, all_entities_name))
            all_entities_offsets = [e.attrib["charOffset"] for e in all_entities]
            all_entities_name_offsets = dict(zip(all_entities_name, all_entities_offsets))
            entity_combinations = create_all_entity_pairs(all_entities_id)
            try:
                all_interactions = sentence.findall("interaction")
            except:
                print("No interaction: ", sentence)
                continue

            positive_interactions = []
            for interaction in all_interactions:
                e1 = interaction.attrib["e1"]  # id of the entity1
                e2 = interaction.attrib["e2"]  # id of the entity2
                e1_name = all_entities_id_names[e1]
                e2_name = all_entities_id_names[e2]
                positive_interactions.append([e1, e2])
                # protein tagging
                e1_offsets = get_entity_offsets(all_entities_name_offsets, e1_name)
                e2_offsets = get_entity_offsets(all_entities_name_offsets, e2_name)
                tagged_sentence = tag_sentence_with_protein(passage=sentence.attrib["text"], e1_offset_list=e1_offsets,
                                                            e2_offset_list=e2_offsets, e1_name=e1_name, e2_name=e2_name)
                # save to df
                df = df.append({'docId': doc.attrib["id"], 'isValid': True, 'passage': tagged_sentence,
                                'passage_id': sentence.attrib["id"]},
                               ignore_index=True)

            # get negative interaction pairs
            negative_interaction_pairs = [p for p in entity_combinations if p not in positive_interactions]
            for interaction in negative_interaction_pairs:
                e1 = interaction[0]  # id of the entity1
                e2 = interaction[1]  # id of the entity2
                e1_name = all_entities_id_names[e1]
                e2_name = all_entities_id_names[e2]
                # protein tagging
                e1_offsets = get_entity_offsets(all_entities_name_offsets, e1_name)
                e2_offsets = get_entity_offsets(all_entities_name_offsets, e2_name)
                tagged_sentence = tag_sentence_with_protein(passage=sentence.attrib["text"], e1_offset_list=e1_offsets,
                                                            e2_offset_list=e2_offsets, e1_name=e1_name, e2_name=e2_name)
                # save to df
                df = df.append({'docId': doc.attrib["id"], 'isValid': False, 'passage': tagged_sentence,
                                'passage_id': sentence.attrib["id"]},
                               ignore_index=True)
    return df



if __name__ == '__main__':

    in_data_dir = 'files'
    out_data_dir = 'files\\out_files'

    input_file_list = ["AIMED"]  # , "IEPA","HPRD50", "BIOINFER", "AIMED"
    for folder_name in input_file_list:
        test_data_df = read_xml(folder_name, split="test")
        test_data_df.to_csv(os.path.join(os.getcwd(), folder_name + "\\out_files\\" + folder_name + "-test.csv"), encoding='utf-8')

        train_data_df = read_xml(folder_name, split="train")
        train_data_df.to_csv(os.path.join(os.getcwd(), folder_name + "\\out_files\\" + folder_name + "-train.csv"), encoding='utf-8')
