
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup

import time
import datetime
import numpy as np


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def flat_accuracy(preds, labels):
    # TODO - what metrics?  MAP?  MSE? etc.

    return 0.0


def build_data_loader(train_tensor, validation_tensor, batch_size):
    #TODO add attension masks

    # Create the DataLoader for our training set.
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_tensor, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set.
    validation_sampler = SequentialSampler(validation_tensor)
    validation_dataloader = DataLoader(validation_tensor, sampler=validation_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader


def train_bert_relevance_model(model, train_dataloader, validation_dataloader, epochs=5, lr=5e-5, eps=1e-8, seed=None):
    # Set the seed value all over the place to make this reproducible.
    # TODO - GPU vs. CPU
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


    optimizer = AdamW(model.parameters(),
                      lr=lr,  # args.learning_rate - default is 5e-5
                      eps=eps  # args.adam_epsilon  - default is 1e-8.
                      )

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

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

            b_input_ids = batch[0] #.to(device)
            b_labels = batch[1] #.to(device)
            #print(b_input_ids)
            #print(b_labels)

            model.zero_grad()

            outputs = model(b_input_ids, labels=b_labels)
            #print(outputs)

            loss = outputs[0]
            #print(loss)

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

            batch = tuple(t for t in batch) #.to(device)

            b_input_ids, b_labels = batch

            with torch.no_grad():

                outputs = model(input_ids=b_input_ids, labels=b_labels)

            logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

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






