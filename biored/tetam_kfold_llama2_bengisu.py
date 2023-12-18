import datetime
import numpy as np
import pandas as pd
import random
import re
import time
import torch

from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)

from trl import SFTTrainer
import re


### HELPER METHODS
###################

# Function to convert seconds to datetime format hh:mm:ss
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def prepare_input_ids_and_attention_masks(tokenizer, sentences, max_len):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        )

        input_ids.append(encoded_sent)

    # print('ex. TOKEN IDS with SPEC:', input_ids[0])
    # print('\nMax sentence length: ', max([len(sen) for sen in input_ids]))
    # print('Padding/truncating all sentences to %d values...' % MAX_LEN)
    # print('Padding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=max_len,
                              dtype="long", truncating="post", padding="post")

    # print('\nex. AFTER PADDING IDS:', input_ids[0])

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

        # print('\nex. ATTENTION MASK:', attention_masks[0])

    return input_ids, attention_masks


def prepare_data_loaders(input_ids, attention_masks, labels, batch_size):
    # Convert to tensors.
    data_inputs = torch.tensor(input_ids)
    data_masks = torch.tensor(attention_masks)
    data_labels = torch.tensor(labels)

    # Create the DataLoader.
    tensor_data = TensorDataset(data_inputs, data_masks, data_labels)
    data_sampler = SequentialSampler(tensor_data)
    data_loader = DataLoader(tensor_data, sampler=data_sampler, batch_size=batch_size)
    return data_loader


def train_model(sentences, labels, fold_num, annotate_ino_terms, add_ino_ids_to_tokenizer):
    ### LOAD PRE_TRAINED BIOBERT_V1.1_PUBMED MODEL AND TOKENIZER FROM HUGGINGFACE-TRANSFORMERS
    ####################
    tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed", do_lower_case=True)
    model = AutoModelForSequenceClassification.from_pretrained("monologg/biobert_v1.1_pubmed")


    ### LOAD  PRETRAINED BIOGPT MODEL & TOKENIZER
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
    # model = AutoModelForSequenceClassification.from_pretrained("microsoft/biogpt")

    # with epoch = 6
    ### HYPER-PARAMETERS
    ####################
    MAX_LEN = 128
    BATCH_SIZE = 2
    EPOCHS = 6
    optimizer = AdamW(model.parameters(),
                      lr=5e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-6,  # args.adam_epsilon  - default is 1e-8.
                      weight_decay=0.1
                      )

    # # with epoch = 4
    ### HYPER-PARAMETERS
    #     MAX_LEN = 128
    #     BATCH_SIZE = 16
    #     EPOCHS = 4
    #     ####################
    #     optimizer = AdamW(model.parameters(),
    #                       lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
    #                       eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
    #                       weight_decay=0.01
    #                     )

    if annotate_ino_terms:
        tokenizer.add_tokens(['[INT]', '[/INT]'], special_tokens=True)

    if add_ino_ids_to_tokenizer:
        INTERACTION_TERMS = 'https://raw.githubusercontent.com/metalrt/protein-interaction-terms/main/interaction_terms.txt'
        interaction_terms_df = pd.read_csv(INTERACTION_TERMS, sep='\t')
        interaction_terms_df.term = interaction_terms_df.term.str.replace('-', '')
        tokenizer.add_tokens(list(set(interaction_terms_df.type.values)))

    model.resize_token_embeddings(len(tokenizer))

    ### USE CUDA AND GPU.
    ####################
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    model.cuda()
    ####################

    ### PREPARE DATALOADERS
    ####################
    input_ids, attention_masks = prepare_input_ids_and_attention_masks(tokenizer, sentences, MAX_LEN)

    # Use 90% for training and 10% for validation.
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                        random_state=2018,
                                                                                        test_size=0.1)
    # Do the same for the masks.
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, random_state=2018, test_size=0.1)

    train_dataloader = prepare_data_loaders(train_inputs, train_masks, train_labels, BATCH_SIZE)
    validation_dataloader = prepare_data_loaders(validation_inputs, validation_masks, validation_labels, BATCH_SIZE)
    #########################

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * EPOCHS

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, EPOCHS):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Fold {:} - Epoch {:} / {:} ========'.format(fold_num, epoch_i + 1, EPOCHS))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels

            batch[2] = batch[2].to(torch.int64)
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out_files of the tuple.
            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have
                # not provided labels.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Training complete!")
    return model, tokenizer, BATCH_SIZE, MAX_LEN


def test_model(model, tokenizer, sentences, labels, fold_num, batch_size, max_len):
    ### USE CUDA AND GPU.
    ####################
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    model.cuda()
    ####################

    ### PREPARE DATALOADERS
    ####################
    input_ids, attention_masks = prepare_input_ids_and_attention_masks(tokenizer, sentences, max_len)

    prediction_dataloader = prepare_data_loaders(input_ids, attention_masks, labels, batch_size)
    #########################

    # Prediction on test set
    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []
    start = time.time()
    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    end = time.time()
    print("prediction_execution_time =", end - start)
    print('    DONE.')

    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    cm = confusion_matrix(flat_true_labels, flat_predictions, labels=[0, 1])

    f1_weighted = f1_score(flat_true_labels, flat_predictions, average='weighted')
    f1_macro = f1_score(flat_true_labels, flat_predictions, average='macro')
    f1_micro = f1_score(flat_true_labels, flat_predictions, average='micro')

    print(f'f1: {f1_score(flat_true_labels, flat_predictions)}')
    print(f'Precision: {precision_score(flat_true_labels, flat_predictions)}')
    print(f'Recall: {recall_score(flat_true_labels, flat_predictions)}')
    print(f'f1_macro: {f1_macro}')
    print(f'macro-Recall: {recall_score(flat_true_labels, flat_predictions, average="macro")}')
    print(f'macro-Precision: {precision_score(flat_true_labels, flat_predictions, average="macro")}')
    print(f'f1_micro: {f1_micro}')
    print(f'micro-Recall: {recall_score(flat_true_labels, flat_predictions, average="micro")}')
    print(f'micro-Precision: {precision_score(flat_true_labels, flat_predictions, average="micro")}')
    print(f'TN: {cm[0][0]}')
    print(f'FN: {cm[1][0]}')
    print(f'TP: {cm[1][1]}')
    print(f'FP: {cm[0][1]}')
    print(f'predictions: {flat_predictions.tolist()}')

    return flat_predictions, precision_score(flat_true_labels, flat_predictions), recall_score(flat_true_labels,
                                                                                               flat_predictions), f1_score(
        flat_true_labels, flat_predictions), precision_score(flat_true_labels, flat_predictions,
                                                             average="macro"), recall_score(flat_true_labels,
                                                                                            flat_predictions,
                                                                                            average="macro"), f1_score(
        flat_true_labels, flat_predictions, average="macro"), precision_score(flat_true_labels, flat_predictions,
                                                                              average="micro"), recall_score(
        flat_true_labels, flat_predictions, average="micro"), f1_score(flat_true_labels, flat_predictions,
                                                                       average="micro")


def prepare_dataset(corpus_name, annotate_ino_terms, use_ino_ids, use_normalized_ino_words):
    if corpus_name == 'MERGED':
        TRAINING_DATA_URL = 'https://raw.githubusercontent.com/metalrt/ppi-dataset/master/csv_output/merged-train.csv'
        TEST_DATA_URL = 'https://raw.githubusercontent.com/metalrt/ppi-dataset/master/csv_output/merged-test.csv'
    elif corpus_name == 'AIMED':
        TRAINING_DATA_URL = 'https://raw.githubusercontent.com/metalrt/ppi-dataset/master/csv_output/AIMed-train.csv'
        TEST_DATA_URL = 'https://raw.githubusercontent.com/metalrt/ppi-dataset/master/csv_output/AIMed-test.csv'
    elif corpus_name == 'BIOINFER':
        TRAINING_DATA_URL = 'https://raw.githubusercontent.com/metalrt/ppi-dataset/master/csv_output/BioInfer-train.csv'
        TEST_DATA_URL = 'https://raw.githubusercontent.com/metalrt/ppi-dataset/master/csv_output/BioInfer-test.csv'
    elif corpus_name == 'HPRD50':
        TRAINING_DATA_URL = 'https://raw.githubusercontent.com/metalrt/ppi-dataset/master/csv_output/HPRD50-train.csv'
        TEST_DATA_URL = 'https://raw.githubusercontent.com/metalrt/ppi-dataset/master/csv_output/HPRD50-test.csv'
    elif corpus_name == 'IEPA':
        TRAINING_DATA_URL = 'https://raw.githubusercontent.com/metalrt/ppi-dataset/master/csv_output/IEPA-train.csv'
        TEST_DATA_URL = 'https://raw.githubusercontent.com/metalrt/ppi-dataset/master/csv_output/IEPA-test.csv'
    elif corpus_name == 'LLL':
        TRAINING_DATA_URL = 'https://raw.githubusercontent.com/metalrt/ppi-dataset/master/csv_output/LLL-train.csv'
        TEST_DATA_URL = 'https://raw.githubusercontent.com/metalrt/ppi-dataset/master/csv_output/LLL-test.csv'
    else:
        print("THERE IS A PROBLEM WITH THE INPUT CORPUS NAME")
        return

    INTERACTION_TERMS = 'https://raw.githubusercontent.com/metalrt/protein-interaction-terms/main/interaction_terms.txt'
    interaction_terms_df = pd.read_csv(INTERACTION_TERMS, sep='\t')
    interaction_terms_df.term = interaction_terms_df.term.str.replace('-', '')
    interactions_dict = dict.fromkeys(interaction_terms_df.term.unique(), 0)
    interaction_term_and_type_dict = dict(zip(interaction_terms_df.term, interaction_terms_df.type))
    interaction_type_and_term_dict = dict()
    for i in range(len(interaction_terms_df)):
        if not interaction_terms_df.type.values[i] in interaction_type_and_term_dict:
            interaction_type_and_term_dict[interaction_terms_df.type.values[i]] = interaction_terms_df.term.values[i]

    ### LOAD DATA INTO CSV.
    ####################
    train_df = pd.read_csv(TRAINING_DATA_URL)
    test_df = pd.read_csv(TEST_DATA_URL)
    df = pd.concat([train_df, test_df], axis=0)
    INSTRUCTION = "In the sentence, if there is a relation between PROTEIN 1 and PROTEIN 2 label as TRUE else label as FALSE."
    for i in range(len(df)):
        df.loc[i, "text"] = "###Instruction:\n{Instruction}\n\n###Sentence:\n{Sentence}\n\n###Label:\n{Label}".format(
            Instruction=INSTRUCTION, Sentence=df.loc[i, "passage"], Label=df.loc[i, "isValid"])

    preprocessed_sentences = list()
    for sentence in df.passage.values:
        interaction_words = list()
        for key in interactions_dict.keys():
            if ' ' + key + ' ' in sentence:
                interaction_words.append(key)
        interaction_words.sort(key=len, reverse=True)
        preprocessed_sentence = sentence
        if annotate_ino_terms:
            for ino_term in interaction_words:
                preprocessed_sentence = re.sub(r'\b' + re.escape(ino_term) + r'\b', '[INT] ' + ino_term + ' [/INT]',
                                               preprocessed_sentence, flags=re.IGNORECASE)
        if use_normalized_ino_words:
            for ino_term in interaction_words:
                preprocessed_sentence = re.sub(r'\b' + re.escape(ino_term) + r'\b',
                                               interaction_type_and_term_dict[interaction_term_and_type_dict[ino_term]],
                                               preprocessed_sentence, flags=re.IGNORECASE)
        if use_ino_ids:
            for ino_term in interaction_words:
                preprocessed_sentence = re.sub(r'\b' + re.escape(ino_term) + r'\b',
                                               interaction_term_and_type_dict[ino_term], preprocessed_sentence,
                                               flags=re.IGNORECASE)
        preprocessed_sentences.append(preprocessed_sentence)

    df['PreProcessedSent'] = preprocessed_sentences

    # df = df.drop_duplicates(subset=['PreProcessedSent'])
    # df = df[df.passage.str.contains("PROTEIN1") & df.passage.str.contains("PROTEIN2")]

    sentences = df.PreProcessedSent.values
    labels = df.isValid.values.astype(int)
    ####################

    # Report the number of sentences.
    print('Number of sentences: {:,}\n'.format(df.shape[0]))

    return df

if __name__ == '__main__':


    all_data = prepare_dataset(corpus_name=corpus_name, annotate_ino_terms=annotate_ino_terms, use_ino_ids=use_ino_ids,
                               use_normalized_ino_words=use_normalized_ino_words)

    corpus_name_in_url = corpus_name
    if corpus_name == "AIMED":
        corpus_name_in_url = "AIMed"
    elif corpus_name == "BIOINFER":
        corpus_name_in_url = "BioInfer"

    # binary
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # macro
    precision_scores_macro = []
    recall_scores_macro = []
    f1_scores_macro = []

    # micro
    precision_scores_micro = []
    recall_scores_micro = []
    f1_scores_micro = []

    document_ids = all_data.docid.unique()

    print("all_data_shape: ", all_data.shape)
    print("all_data_len: ", len(all_data))
    print("document_ids_len", len(document_ids))

    for fold_num in range(1, 11):
        val_docids = pd.read_csv(
            "https://raw.githubusercontent.com/metalrt/ppi-cv-splits/main/" + corpus_name_in_url + "/" + corpus_name_in_url + str(
                fold_num) + ".txt", header=None)
        val_docids = val_docids[0]

        train_df = all_data[~all_data['docid'].isin(val_docids)]
        val_df = all_data[all_data['docid'].isin(val_docids)]

        # train the model
        train_sentences = train_df.PreProcessedSent.values
        train_labels = train_df.isValid.values.astype(int)
        model, tokenizer, batch_size, max_len = train_model(sentences=train_sentences, labels=train_labels,
                                                            fold_num=fold_num, annotate_ino_terms=annotate_ino_terms,
                                                            add_ino_ids_to_tokenizer=add_ino_ids_to_tokenizer)

        # test the model
        test_sentences = val_df.PreProcessedSent.values
        test_labels = val_df.isValid.values.astype(int)
        flat_predictions, precision, recall, f1, precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro = test_model(
            model=model, tokenizer=tokenizer, sentences=test_sentences, labels=test_labels, fold_num=fold_num,
            batch_size=batch_size, max_len=max_len)

        # append model score binary
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        # append model score macro
        precision_scores_macro.append(precision_macro)
        recall_scores_macro.append(recall_macro)
        f1_scores_macro.append(f1_macro)

        # append model score micro
        precision_scores_micro.append(precision_micro)
        recall_scores_micro.append(recall_micro)
        f1_scores_micro.append(f1_micro)

        del model
        torch.cuda.empty_cache()

    print("f1_scores", f1_scores)
    print(f"Mean-f1: {sum(f1_scores) / len(f1_scores)}")
    return precision_scores, recall_scores, f1_scores, precision_scores_macro, recall_scores_macro, f1_scores_macro, precision_scores_micro, recall_scores_micro, f1_scores_micro

