import torch
from torch.autograd import Variable


class BatchPredictor(object):

    def __init__(self, model, src_vocab, tgt_vocab):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def get_decoder_features(self, src_seqs):
        # src_id_seq = torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]).view(1, -1)
        max_length = 0 
        for seq in src_seqs:
            if len(seq) >= max_length:
                max_length = len(seq)
                
        src_id_seqs = []
        for seq in src_seqs:
            padding = [0 for _ in range(max_length-len(seq))]
            s = [self.src_vocab.stoi[tok] for tok in seq]
            s.extend(padding)
            src_id_seqs.append(torch.LongTensor(s))
        
        src_id_seqs = torch.stack(src_id_seqs)

        ll = [len(seq) for seq in src_seqs]
        
        if torch.cuda.is_available():
            src_id_seqs = src_id_seqs.cuda()

        with torch.no_grad():
            # softmax_list, _, other = self.model(src_id_seq, [len(src_seq)])
            # print(src_id_seqs)
            # exit()
            softmax_list, _, other = self.model(src_id_seqs, ll)

        return other

    def encode(self, src_seqs):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        # other = self.get_decoder_features(src_seq)

        # length = other['length'][0]
        # encoding = other['representation']

        # # tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        # # tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]

        # return encoding, encoding.shape
        max_length = 0 
        for seq in src_seqs:
            if len(seq) >= max_length:
                max_length = len(seq)
                
        src_id_seqs = []
        for seq in src_seqs:
            padding = [0 for _ in range(max_length-len(seq))]
            # s = [self.src_vocab.stoi[tok] for tok in seq]
            s = [self.src_vocab.stoi[tok] if tok in self.src_vocab.stoi else self.src_vocab.stoi['<unk>'] for tok in seq]
            s.extend(padding)
            src_id_seqs.append(torch.LongTensor(s))
        
        src_id_seqs = torch.stack(src_id_seqs)
        
        lengths = [len(seq) for seq in src_seqs]
    
        if torch.cuda.is_available():
            src_id_seqs = src_id_seqs.cuda()

        with torch.no_grad():
            _, encoder_hidden = self.model.encoder(src_id_seqs, lengths)

        return encoder_hidden, encoder_hidden.shape
    
    def predict(self, src_seqs):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        other = self.get_decoder_features(src_seqs)

        # length = other['length'][0]
        lengths = other['length']
        
        # tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        
        tgt_seqs = []
        
        for j in range(len(src_seqs)):
            tgt_id_seq = [other['sequence'][di][j].data[0] for di in range(lengths[j])]
            tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
            tgt_seqs.append(tgt_seq)
            
        # tgt_id_seqs = [[other['sequence'][di][0].data[0] for di in range(length)] for length in lengths]

        # tgt_seq = [[self.tgt_vocab.itos[tok] for tok in tgt_id_seq] for tgt_id_seq in tgt_id_seqs]

        return tgt_seqs

    def predict_n(self, src_seq, n=1):
        """ Make 'n' predictions given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language
            n (int): number of predicted seqs to return. If None,
                     it will return just one seq.

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
                            by the pre-trained model
        """
        other = self.get_decoder_features(src_seq)

        result = []
        for x in range(0, int(n)):
            length = other['topk_length'][0][x]
            tgt_id_seq = [other['topk_sequence'][di][0, x, 0].data[0] for di in range(length)]
            tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
            result.append(tgt_seq)

        return result
