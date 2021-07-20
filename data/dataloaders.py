import torch

class SemanticSearchDataCollator:
    """Collates data for semanatic search, uses differing tokenizers for code and text. Returns a dictionary of tensors
    each corresponding to a batch a cross a feature.
    """

    def __init__(self, code_tokenizer, doc_tokenizer):
        """
        :param code_tokenizer: the tokenizer for the code model
        :param doc_tokenizer: the tokenizer for the docstring model
        """
        self.code_tokenizer = code_tokenizer
        self.doc_tokenizer = doc_tokenizer
        # pytorch ignores -100 in the loss calculation so we use it here to pad the labels
        self.pad_token = int(-100)

    def __call__(self, features):
        """
        :param features: a batch of features passed from the Dataset object, a list of dicitoinaries where each list
        element is a datapoint and the dictionary keys point to features of that datapoint
        :return: a dictionary of tensors to be passed to the model for training
        """
        # first we cast encoder/decoder input ids to lists so we can easily identify the longest one
        # todo think of a better wasy to do this as we loop through the data once here and then again to calculate the max

        encoder_input_ids = [feature["input_ids"] for feature in features] if "input_ids" in features[
            0].keys() else None
        decoder_input_ids = [feature["decoder_input_ids"] for feature in features] if "decoder_input_ids" in features[
            0].keys() else None

        # then we define placeholders to hold the data whilst we assemble the tensors
        out_input_ids = []
        out_input_mask = []
        out_decoder_ids = []
        out_attention_mask = []
        out_labels = []

        # we calculate the maximum length of encoder/decoder inputs in the current batch
        enc_max_length = max(len(l) for l in encoder_input_ids)
        dec_max_length = max(len(l) for l in decoder_input_ids)

       # we pad each datapoint in the batch to this maximum length, treating the encoder and decoderr inputs separatley
        for feature in features:

            # as 1 is the id for the pad token for the Roberta model we pad with ones to max length of batch
            enc_remainder = torch.ones((enc_max_length - len(feature["input_ids"])), dtype=torch.int64)
            enc_attention_remainder = torch.zeros(len(enc_remainder), dtype=torch.int64)

            # at inference time we don't have access to decoder inputs
            if decoder_input_ids:
                dec_remainder = torch.ones((dec_max_length - len(feature["decoder_input_ids"])),
                                           dtype=torch.int64)
                dec_attention_remainder = torch.zeros(len(dec_remainder), dtype=torch.int64)
                label_padding = torch.ones((dec_max_length - len(feature["decoder_input_ids"])), dtype=torch.int64) * int(
                    -100)

            out_input_ids.append(torch.cat((feature["input_ids"], enc_remainder), 0) )
            out_input_mask.append(torch.cat((feature["attention_mask"], enc_attention_remainder), 0))

            if decoder_input_ids:
                out_decoder_ids.append(torch.cat((feature["decoder_input_ids"], dec_remainder), 0))
                out_attention_mask.append(torch.cat((feature["decoder_attention_mask"], dec_attention_remainder), 0))
                out_labels.append(torch.cat((feature["decoder_input_ids"], label_padding), 0))


        input_id_tensor = torch.stack(out_input_ids)
        input_attention_mask_tensor = torch.stack(out_input_mask)

        if decoder_input_ids:
            decoder_id_tensor = torch.stack(out_decoder_ids)
            decoder_attention_mask_tensor = torch.stack(out_attention_mask)
            label_tensor = torch.stack(out_labels)

        # we assemble the dictionary of tensors to be passed to the seq2seq model
        if decoder_input_ids:
            output = {
                "input_ids": input_id_tensor,
                "attention_mask": input_attention_mask_tensor,
                "decoder_input_ids": decoder_id_tensor,
                "decoder_attention_mask": decoder_attention_mask_tensor,
                "labels": label_tensor
            }
        else:
            output = {
                "input_ids": input_id_tensor,
                "attention_mask": input_attention_mask_tensor
            }

        return output

        # if decoder_input_ids:
        #     enc_max_length = max(len(l) for l in encoder_input_ids)
        #     dec_max_length = max(len(l) for l in decoder_input_ids)
        #     for feature in features:
        #         # as 1 is the id for the pad token for the Roberta model we pad with ones to max length of batch
        #         enc_remainder = torch.ones((enc_max_length - len(feature["input_ids"])),
        #                                    dtype=torch.int64)  # [1] * (max_length - len(feature["decoder_input_ids"]))
        #         enc_attention_remainder = torch.zeros(len(enc_remainder), dtype=torch.int64)
        #
        #         out_input_ids.append(torch.cat((feature["input_ids"], enc_remainder), 0))
        #         out_input_mask.append(torch.cat((feature["attention_mask"], enc_attention_remainder), 0))
        #
        # if decoder_input_ids:
        #     max_length = max(len(l) for l in decoder_input_ids)
        #     for feature in features:
        #         # as 1 is the id for the pad token for the Roberta model we pad with ones to max length of batch
        #         dec_remainder = torch.ones((max_length - len(feature["decoder_input_ids"])),
        #                                    dtype=torch.int64)  # [1] * (max_length - len(feature["decoder_input_ids"]))
        #         label_padding = torch.ones((max_length - len(feature["decoder_input_ids"])), dtype=torch.int64) * int(
        #             -100)
        #         dec_attention_remainder = torch.zeros(len(dec_remainder), dtype=torch.int64)
        #         out_decoder_ids.append(torch.cat((feature["decoder_input_ids"], dec_remainder), 0))
        #         out_attention_mask.append(torch.cat((feature["decoder_attention_mask"], dec_attention_remainder), 0))
        #         out_labels.append(torch.cat((feature["decoder_input_ids"], label_padding), 0))

    # if decoder_input_ids:
    #     max_length = max(len(l) for l in decoder_input_ids)
    #     for feature in features:
    #         # as 1 is the id for the pad token for the Roberta model we pad with ones to max length of batch
    #         dec_remainder = torch.ones((max_length - len(feature["decoder_input_ids"])),
    #                                    dtype=torch.int64)  # [1] * (max_length - len(feature["decoder_input_ids"]))
    #         label_padding = torch.ones((max_length - len(feature["decoder_input_ids"])), dtype=torch.int64) * int(-100)
    #         dec_attention_remainder = torch.zeros(len(dec_remainder), dtype=torch.int64)
    #         out_decoder_ids.append(torch.cat((feature["decoder_input_ids"], dec_remainder), 0))
    #         out_attention_mask.append(torch.cat((feature["decoder_attention_mask"], dec_attention_remainder), 0))
    #         out_labels.append(torch.cat((feature["decoder_input_ids"], label_padding), 0))
