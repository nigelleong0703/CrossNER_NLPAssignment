import torch
import torch.nn as nn

from src.conll2002_metrics import *
from src.dataloader import (
    domain2labels,
    pad_token_label_id,
    span_labels,
    domain2labels_only,
)

import os
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger()


class BertSpanTypeTrainer(object):
    def __init__(self, params, span_model, type_model):
        self.params = params
        self.span_model = span_model
        self.type_model = type_model

        self.optimizer = torch.optim.Adam(
            list(self.span_model.parameters()) + list(self.type_model.parameters()),
            lr=params.lr,
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.early_stop = params.early_stop
        self.no_inprovement_num = 0
        self.best_acc = 0

    def train_step(self, X, labels_bio, y, real_Y):
        """
        need to return: loss, loss_span and loss_type
        """
        self.span_model.train()
        self.type_model.train()
        output_span = self.span_model(X)
        # output_span[0] logits, output_span[1] final_embedding
        labels_bio_flat = labels_bio.view(labels_bio.size(0) * labels_bio.size(1))
        output_span_flat = output_span[0].view(
            output_span[0].size(0) * output_span[0].size(1), output_span[0].size(2)
        )
        loss_span = self.loss_fn(output_span_flat, labels_bio_flat)

        # output_type = self.type_model(X, output_span[0]) # train with predicted span
        output_type = self.type_model(X)  # train with ground truth
        # output_type[0] logits, output_type[1] final_embedding
        y_flat = y.view(y.size(0) * y.size(1))
        output_type_flat = output_type[0].view(
            output_type[0].size(0) * output_type[0].size(1), output_type[0].size(2)
        )
        loss_type = self.loss_fn(output_type_flat, y_flat)

        self.optimizer.zero_grad()
        loss = loss_span + loss_type
        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_span.item(), loss_type.item()

    def evaluate(self, dataloader, tgt_dm, use_bilstm=False):
        self.span_model.eval()
        self.type_model.eval()
        pred_list = []
        y_list = []
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (X, span, Y, real_Y) in pbar:
            y_list.extend(real_Y.data.numpy())
            X = X.cuda()
            span = span.cuda()
            Y = Y.cuda()
            real_Y = real_Y.cuda()

            output_span = self.span_model(X)
            output_type = self.type_model(X, output_span[0])

            # Convert logits to categorical predictions
            span_predictions = torch.argmax(
                output_span[0], dim=-1
            )  # Assuming output_span is [batch_size, seq_length, num_span_labels]
            type_predictions = torch.argmax(
                output_type[0], dim=-1
            )  # Assuming output_type is [batch_size, seq_length, num_type_labels]

            combined_indices = []

            for i in range(span_predictions.size(0)):  # Loop through batch
                combined_seq_indices = []
                for j in range(
                    span_predictions.size(1)
                ):  # Loop through sequence length
                    span_pred = span_labels[span_predictions[i, j].item()]

                    type_pred_index = type_predictions[i, j].item()
                    if type_pred_index >= len(domain2labels_only[tgt_dm]):
                        type_pred_index = (
                            len(domain2labels_only[tgt_dm]) - 1
                        )  # Use last index if out of bounds
                    type_pred = domain2labels_only[tgt_dm][type_pred_index]
                    combined_label = (
                        "O"
                        if span_pred == "O" or type_pred == "O"
                        else f"{span_pred}-{type_pred}"
                    )
                    combined_seq_indices.append(
                        domain2labels[tgt_dm].index(combined_label)
                    )

                combined_indices.append(combined_seq_indices)

            # pred_list.extend(combined_indices.numpy())
            pred_list.extend(np.array(combined_indices))

        pred_list = np.concatenate(pred_list, axis=0)
        # pred_list = np.argmax(pred_list, axis=1)
        y_list = np.concatenate(y_list, axis=0)

        # calculate f1 score
        pred_list = list(pred_list)
        y_list = list(y_list)
        lines = []
        for pred_index, gold_index in zip(pred_list, y_list):
            gold_index = int(gold_index)
            if gold_index != pad_token_label_id:
                pred_token = domain2labels[tgt_dm][pred_index]
                gold_token = domain2labels[tgt_dm][gold_index]
                lines.append("w" + " " + pred_token + " " + gold_token)
        results = conll2002_measure(lines)
        f1 = results["fb1"]

        return f1

    def save_model(self):
        saved_path = os.path.join(self.params.dump_path, "best_finetune_model.pth")
        torch.save(
            {
                "span_model": self.span_model,
                "type_model": self.type_model,
            },
            saved_path,
        )
        logger.info("Best model has been saved to %s" % saved_path)

    def train_conll(self, dataloader_train, dataloader_dev, dataloader_test, tgt_dm):
        logger.info("Pretraining on conll2003 NER dataset ...")
        no_improvement_num = 0
        best_f1 = 0
        for e in range(self.params.epoch):
            logger.info("============== epoch %d ==============" % e)
            loss_list = []
            loss_span_list = []
            loss_type_list = []

            pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
            for i, (X, labels_bio, y, real_y) in pbar:
                X, labels_bio, y = X.cuda(), labels_bio.cuda(), y.cuda()
                real_y = real_y.cuda()

                loss, loss_span, loss_type = self.train_step(X, labels_bio, y, real_y)
                loss_list.append(loss)
                loss_span_list.append(loss_span)
                loss_type_list.append(loss_type)
                pbar.set_description(
                    "(Epoch {}) LOSS:{:.4f} LOSS_SPAN:{:.4f} LOSS_TYPE:{:.4f}".format(
                        e,
                        np.mean(loss_list),
                        np.mean(loss_span_list),
                        np.mean(loss_type_list),
                    )
                )
                # pbar.set_description(
                #     "(Epoch {}) LOSS:{:.4f}".format(e, np.mean(loss_list))
                # )

            logger.info(
                "Finish training epoch %d. loss: %.4f" % (e, np.mean(loss_list))
            )

            logger.info(
                "============== Evaluate epoch %d on Dev Set ==============" % e
            )
            f1_dev = self.evaluate(dataloader_dev, tgt_dm)
            logger.info("Evaluate on Dev Set. F1: %.4f." % f1_dev)

            if f1_dev > best_f1:
                logger.info("Found better model!!")
                best_f1 = f1_dev
                no_improvement_num = 0
            else:
                no_improvement_num += 1
                logger.info("No better model found (%d/%d)" % (no_improvement_num, 1))

            # if no_improvement_num >= 1:
            #     break
            if e >= 1:
                break

        logger.info("============== Evaluate on Test Set ==============")
        f1_test = self.evaluate(dataloader_test, tgt_dm)
        logger.info("Evaluate on Test Set. F1: %.4f." % f1_test)


class BaseTrainer(object):
    def __init__(self, params, model):
        self.params = params
        self.model = model

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)
        self.loss_fn = nn.CrossEntropyLoss()

        self.early_stop = params.early_stop
        self.no_improvement_num = 0
        self.best_acc = 0

    def train_step(self, X, y):
        self.model.train()

        preds = self.model(X)
        y = y.view(y.size(0) * y.size(1))  # (bsz, seq_len) -> (bsz * seq_len)
        preds = preds.view(
            preds.size(0) * preds.size(1), preds.size(2)
        )  # (bsz, seq_len, num_tag) -> (bsz * seq_len, num_tag)

        self.optimizer.zero_grad()
        loss = self.loss_fn(preds, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_step_for_bilstm(self, X, lengths, y):
        self.model.train()
        preds = self.model(X)
        loss = self.model.crf_loss(preds, lengths, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, dataloader, tgt_dm, use_bilstm=False):
        self.model.eval()

        pred_list = []
        y_list = []
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        if use_bilstm:
            for i, (X, lengths, y) in pbar:
                y_list.extend(y)
                X, lengths = X.cuda(), lengths.cuda()
                preds = self.model(X)
                preds = self.model.crf_decode(preds, lengths)
                pred_list.extend(preds)
        else:
            for i, (X, y) in pbar:
                y_list.extend(y.data.numpy())  # y is a list
                X = X.cuda()
                preds = self.model(X)
                pred_list.extend(preds.data.cpu().numpy())

        # concatenation
        pred_list = np.concatenate(pred_list, axis=0)  # (length, num_tag)
        if not use_bilstm:
            pred_list = np.argmax(pred_list, axis=1)
        y_list = np.concatenate(y_list, axis=0)

        # calcuate f1 score
        pred_list = list(pred_list)
        y_list = list(y_list)
        lines = []
        for pred_index, gold_index in zip(pred_list, y_list):
            gold_index = int(gold_index)
            if gold_index != pad_token_label_id:
                pred_token = domain2labels[tgt_dm][pred_index]
                gold_token = domain2labels[tgt_dm][gold_index]
                lines.append("w" + " " + pred_token + " " + gold_token)
        results = conll2002_measure(lines)
        f1 = results["fb1"]

        return f1

    def train_conll(self, dataloader_train, dataloader_dev, dataloader_test, tgt_dm):
        logger.info("Pretraining on conll2003 NER dataset ...")
        no_improvement_num = 0
        best_f1 = 0
        for e in range(self.params.epoch):
            logger.info("============== epoch %d ==============" % e)
            loss_list = []

            pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
            for i, (X, y) in pbar:
                X, y = X.cuda(), y.cuda()

                loss = self.train_step(X, y)
                loss_list.append(loss)
                pbar.set_description(
                    "(Epoch {}) LOSS:{:.4f}".format(e, np.mean(loss_list))
                )

            logger.info(
                "Finish training epoch %d. loss: %.4f" % (e, np.mean(loss_list))
            )

            logger.info(
                "============== Evaluate epoch %d on Dev Set ==============" % e
            )
            f1_dev = self.evaluate(dataloader_dev, tgt_dm)
            logger.info("Evaluate on Dev Set. F1: %.4f." % f1_dev)

            if f1_dev > best_f1:
                logger.info("Found better model!!")
                best_f1 = f1_dev
                no_improvement_num = 0
            else:
                no_improvement_num += 1
                logger.info("No better model found (%d/%d)" % (no_improvement_num, 1))

            # if no_improvement_num >= 1:
            #     break
            if e >= 1:
                break

        logger.info("============== Evaluate on Test Set ==============")
        f1_test = self.evaluate(dataloader_test, tgt_dm)
        logger.info("Evaluate on Test Set. F1: %.4f." % f1_test)

    def save_model(self):
        """
        save the best model
        """
        saved_path = os.path.join(self.params.dump_path, "best_finetune_model.pth")
        torch.save(
            {
                "model": self.model,
            },
            saved_path,
        )
        logger.info("Best model has been saved to %s" % saved_path)
