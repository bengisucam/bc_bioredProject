import json
import os.path

import pandas as pd


def read_json(path, file_name):
    # read the json file
    with open(path + file_name) as train_file:
        dict_train = json.load(train_file)

    dev = pd.DataFrame(dict_train["documents"])
    entity_df = pd.DataFrame(columns=['Identifier', 'Name', 'Type', 'Location', 'From'])
    relation_df = pd.DataFrame(columns=['Entity1', 'Entity2', 'RelationType', 'Novel'])
    for ind in dev.index:
        pmid = dev['id'][ind]
        passage = dev["passages"][ind]
        title = passage[0]["text"]
        title_entities = passage[0]["annotations"]
        abstract = passage[1]["text"]
        abstact_entities = passage[1]["annotations"]
        relation = dev["relations"][ind]

        for e in title_entities:
            identifier = e["infons"]["identifier"]
            type = e["infons"]["type"]
            name = e["text"]
            location = [e["locations"][0]["offset"], e["locations"][0]["length"]]
            entity_df = entity_df.append(
                {'PMID': int(pmid), 'Identifier': identifier, 'Name': name, 'Type': type, 'Location': location,
                 'From': 'Title'}, ignore_index=True)
        for e in abstact_entities:
            identifier = e["infons"]["identifier"]
            type = e["infons"]["type"]
            name = e["text"]
            location = [e["locations"][0]["offset"], e["locations"][0]["length"]]
            entity_df = entity_df.append(
                {'PMID': int(pmid), 'Identifier': identifier, 'Name': name, 'Type': type, 'Location': location,
                 'From': 'Abstract'}, ignore_index=True)
        for r in relation:
            entity1 = r["infons"]["entity1"]
            entity2 = r["infons"]["entity2"]
            relation_type = r["infons"]["type"]
            novel = bool(r["infons"]["novel"] == "Novel")
            relation_df = relation_df.append({'Entity1': entity1, 'Entity2': entity2, 'RelationType': relation_type,
                                              'Novel': novel}, ignore_index=True)

    return entity_df, relation_df


def read_tsv(path, filename):
    dev_tsv = pd.read_csv(path + filename, sep='\t')
    dev_tsv.columns = ['PMID', 'ID1', 'ID2', 'InSentence', 'SentenceWindow',
                       'Document', 'Neighbours', 'RelationType', 'Novelty']
    # dev_tsv["InSentence"] = dev_tsv['InSentence'].astype(str)
    return dev_tsv


def preprocessSentenceAndEntities(entity1_df, entity2_df, doc):
    chars_to_remove = [',', '[', ']', '(', ')', '{', '}', '-']
    E1 = ''
    E2 = ''

    # get the entity names and lower them, put all unique values into list
    entity1_list = list(str.lower(x) for x in set(entity1_df['Name'].to_list()))
    entity2_list = list(str.lower(x) for x in set(entity2_df['Name'].to_list()))
    # lover the sentence and replace the chars to remove with " "
    doc = str.lower(doc)
    for char in chars_to_remove:
        doc = doc.replace(char, " ")
    # remove the double and triple spaces in the sentence
    doc = doc.replace("   ", " ")
    doc = doc.replace("  ", " ")
    # replace the chars to remove with " " for each entity
    for i in range(len(entity1_list)):
        for char in chars_to_remove:
            entity1_list[i] = entity1_list[i].replace(char, " ")
        # remove double and triple spaces from the entities
        entity1_list[i] = entity1_list[i].replace("   ", " ")
        entity1_list[i] = entity1_list[i].replace("  ", " ")
    for i in range(len(entity2_list)):
        for char in chars_to_remove:
            entity2_list[i] = entity2_list[i].replace(char, " ")
        # remove double and triple spaces from the entities
        entity2_list[i] = entity2_list[i].replace("   ", " ")
        entity2_list[i] = entity2_list[i].replace("  ", " ")

    for e1 in entity1_list:
        if e1 in doc:
            E1 = e1
            doc = doc.replace(e1, ' PROTEIN1 ')
    for e2 in entity2_list:
        if e2 in doc:
            E2 = e2
            doc = doc.replace(e2, ' PROTEIN2 ')
    return entity1_list, entity2_list, E1, E2, doc


def find_relation_sentence(document):
    # split the document into a list of sentences
    sentences = document.split('. ')
    # iterate over each sentence in the list
    for sentence in sentences:
        # check if the sentence contains both "here" and "there" words
        if 'PROTEIN1' in sentence and 'PROTEIN2' in sentence:
            # return the sentence if it does contain both words
            return sentence
    # if no sentence contains both "here" and "there" words, return None
    return None


if __name__ == '__main__':

    biored_path = "C://Users//B3LAB//Desktop//thesis//BioRED"

    # read the json file of bioRED
    # dev_entities, dev_relations = read_json(biored_path, '//Dev.BioC.JSON')
    # dev_entities, dev_relations = read_json(biored_path, '//Train.BioC.JSON')
    dev_entities, dev_relations = read_json(biored_path, '//Test.BioC.JSON')

    # read the tsv file created by prepareBioREDBertGT
    # dev_tsv = read_tsv(biored_path, '//processed//dev.tsv')
    #dev_tsv = read_tsv(biored_path, '//processed//train.tsv')
    dev_tsv = read_tsv(biored_path, '//processed//test.tsv')

    # resulting dataframe
    result = pd.DataFrame(
        columns=['SentID', 'PMID', 'Symbol1', 'Symbol2', 'Entity1', 'Entity2', 'OrigSent', 'PreProcessedSent', 'RelationType'])

    # add the respective entity names
    unique_pmids = dev_tsv['PMID'].unique()
    combined_df = pd.DataFrame()
    num_of_total_rel_pairs = 0
    for pmid in unique_pmids:
        doc = dev_tsv[dev_tsv['PMID'] == pmid]
        num_of_total_rel_pairs += doc.shape[0]
        # get the in sentence relations only
        doc_sentence_only = doc[doc['InSentence']]
        # reset the indexins
        doc_sentence_only.reset_index(drop=True, inplace=True)
        # create additional columns except the AllEntities column
        doc_sentence_only['SentID'], doc_sentence_only['OrigSent'], doc_sentence_only[
            'PreProcessedSent'] = doc_sentence_only.index, "", "",
        doc_sentence_only['Entity1'], doc_sentence_only['Entity2'] = '', ''
        # get the entities which have the same pmid with the document
        doc_entities = dev_entities[dev_entities['PMID'] == pmid]
        if not doc_sentence_only.empty:
            for ind in doc_sentence_only.index:
                doc_sentence_only['Symbol1'] = doc_sentence_only['ID1'][ind]
                doc_sentence_only['Symbol2'] = doc_sentence_only['ID2'][ind]
                doc_sentence_only['RelationType'] = doc_sentence_only['RelationType'][ind]
                entity1_df = doc_entities[doc_entities['Identifier'] == doc_sentence_only['ID1'][ind]]
                entity2_df = doc_entities[doc_entities['Identifier'] == doc_sentence_only['ID2'][ind]]
                orig_document = doc_sentence_only["Document"][ind]
                processed_entity1_list, processed_entity2_list, entity1, entity2, processed_document = \
                    preprocessSentenceAndEntities(entity1_df, entity2_df, orig_document)
                processed_sentence = find_relation_sentence(processed_document)
                doc_sentence_only['OrigSent'][ind] = orig_document
                doc_sentence_only['PreProcessedSent'][ind] = processed_sentence
                doc_sentence_only['Entity1'][ind] = entity1
                doc_sentence_only['Entity2'][ind] = entity2

            result = pd.concat([result, doc_sentence_only[['SentID', 'PMID', 'Symbol1', 'Symbol2',
                                                           'Entity1', 'Entity2', 'OrigSent', 'PreProcessedSent', 'RelationType']]],  ignore_index=True)

            result = result.reset_index(drop=True)

    print("Num of In Sentence Relation Pairs: ", result.shape[0])
    print("Num of All Relation Pairs: ", num_of_total_rel_pairs)
    # save the dataframe to a CSV file
    #result.to_csv(biored_path + '//new//dev_biored_sentenceOnly.csv', index=False)
    #result.to_csv(biored_path + '//new//train_biored_sentenceOnly.csv', index=False)
    result.to_csv(biored_path + '//new//test_biored_sentenceOnly.csv', index=False)



