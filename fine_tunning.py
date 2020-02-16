
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
from preprocessing import build_data_loader

import torch
import time
import datetime
import numpy as np
import random


def format_time(elapsed):

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def train_bert_relevance_model(model, train_dataloader, validation_dataloader, epochs=5, lr=5e-5, eps=1e-8, seed_val=42):

    # Set the seed value all over the place to make this reproducible.

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # If there's a GPU available...
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

        model.cuda()

    # If not...
    else:

        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    loss_values = []

    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_token_type_ids = batch[1].to(device)
            b_attention_mask = batch[2].to(device)
            b_labels = batch[3].to(device, dtype=torch.float)

            model.zero_grad()

            outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, token_type_ids=b_token_type_ids,
                            labels=b_labels)

            loss = outputs[0]

            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print(avg_train_loss)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================

        t0 = time.time()

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in validation_dataloader:

            b_input_ids = batch[0].to(device)
            b_token_type_ids = batch[1].to(device)
            b_attention_mask = batch[2].to(device)
            b_labels = batch[3].to(device, dtype=torch.float)

            with torch.no_grad():

                outputs = model(input_ids=b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_attention_mask,
                                labels=b_labels)

            loss = outputs[0]
            print('*** LOSS ***')
            print(loss)

            # print('*** PRED ***')
            # pred = outputs[1].numpy().tolist()
            #
            # print('*** GT ***')
            # gt = b_labels.numpy()
            #gt_list = b_labels.numpy().tolist()
            # print(gt)
            # print(gt_list)

            # Calculate the accuracy for this batch of test sentences.
            #tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            #eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        #print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Training complete!")

    #TODO - trec output wrtiter


if __name__ == "__main__":

    from bert_models import BertForRelevance

    train_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_dataset.pt'
    dev_path = '/nfs/trec_car/data/bert_reranker_datasets/test_dataset.pt'

    train_tensor = torch.load(train_path)
    validation_tensor = torch.load(dev_path)

    train_dataloader, validation_dataloader = build_data_loader(train_tensor=train_tensor,
                                                                validation_tensor=validation_tensor,
                                                                batch_size=32)

    pretrained_weights = 'bert-base-uncased'
    relevance_bert = BertForRelevance.from_pretrained(pretrained_weights)

    train_bert_relevance_model(model=relevance_bert, train_dataloader=train_dataloader,
                               validation_dataloader=validation_dataloader,
                               epochs=5,
                               lr=5e-4,
                               eps=1e-8)








