import re

import torch
from torch_geometric.data import Data

from data.utils.pyg_plot import plot_pyg_graph_with_graphviz
from experiments.holist.utilities.lean_sexpression_to_graph import sexpression_to_graph
from experiments.holist.utilities.sexpression_graphs import SExpressionGraph

if __name__ == '__main__':
    with open('sexp_data_dir/cleaned_training_data/train.src') as f:
        src = f.read().splitlines()

    with open('sexp_data_dir/cleaned_training_data/train.tgt') as f:
        tgt = f.read().splitlines()

    _TOKEN_RE = re.compile(r'\{|[^\s{},;\[\]():\']+')

    # extract first tactic for now for prediction
    first_tacs = []
    for t in tgt:
        toks = _TOKEN_RE.findall(t)
        first_tacs.append(toks[0] if toks[0] != '{' else toks[1])


    # print (src[0])
    print (sexpression_to_graph(src[0]))

    graph = SExpressionGraph(src[0])
    print (graph.to_text(graph.roots()[0]))
    ch = [graph.to_text(a) for a in graph.get_children(graph.to_text(graph.roots()[0]))]
    ch1 =  ([graph.to_text(a) for a in graph.get_children(ch[2])])
    ch2 =  ([graph.to_text(a) for a in graph.get_children(ch1[1])])
    ch3 =  ([graph.to_text(a) for a in graph.get_children(ch2[0])])
    # ch4 =  ([graph.to_text(a) for a in graph.get_children(ch3[0])])
    # ch5 =  ([graph.to_text(a) for a in graph.get_children(ch4[0])])
    # ch6 =  ([graph.to_text(a) for a in graph.get_children(ch5[0])])
    print (ch)
    print (ch1)
    print (ch2)
    print (ch3)
    # print (ch4)
    # print (ch5)
    # print (ch6)


    # src[0] = '(PI (: _inst_2 (nontrivial nat)) (PI (: _inst_3 ((unique_factorization_monoid nat) nat.cancel_comm_monoid_with_zero)) (PI (: _inst_4 ((normalization_monoid nat) nat.cancel_comm_monoid_with_zero)) (PI (: _inst_5 (decidable_eq nat)) (PI (: x nat) (PI (: x0 (((ne nat) x) ((has_zero.zero nat) ((mul_zero_class.to_has_zero nat) ((mul_zero_one_class.to_mul_zero_class nat) ((monoid_with_zero.to_mul_zero_one_class nat) ((comm_monoid_with_zero.to_monoid_with_zero nat) ((cancel_comm_monoid_with_zero.to_comm_monoid_with_zero nat) nat.cancel_comm_monoid_with_zero)))))))) ((iff (((squarefree nat) ((monoid_with_zero.to_monoid nat) ((comm_monoid_with_zero.to_monoid_with_zero nat) ((cancel_comm_monoid_with_zero.to_comm_monoid_with_zero nat) nat.cancel_comm_monoid_with_zero)))) x)) ((multiset.nodup nat) ((((((unique_factorization_monoid.normalized_factors nat) nat.cancel_comm_monoid_with_zero) (LAMBDA (: a nat) (LAMBDA (: b nat) ((_inst_5 a) b)))) _inst_4) _inst_3) x)))))))))'
    print (sexpression_to_graph(src[0]))

    g = sexpression_to_graph(src[0])

    data = Data(torch.arange(len(g['tokens'])), torch.LongTensor(g['edge_index']), torch.LongTensor(g['edge_attr']))
    plot_pyg_graph_with_graphviz(data, g['tokens'])

