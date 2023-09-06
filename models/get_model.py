from models.amr.amr import AMRTransformer
from models.gnn.digae.digae_model import DigaeEmbedding
from models.gnn.formula_net.formula_net import FormulaNet, FormulaNetEdges
from models.gnn.gcn import GCNGNN, DiGCNGNN
from models.gnn_transformer import GNNTransformer
from models.relation_transformer.relation_transformer_new import AttentionRelations
from models.relation_transformer.relation_transformer_small import AttentionRelationSmall
from models.sat.models import GraphTransformer
from models.holist_models.sat.models import GraphTransformer as HOListSAT
from models.tacticzero_autoencoder.tacticzero_autoencoder import TacticZeroAutoEncoder
from models.transformer.transformer_encoder_model import TransformerWrapper
from models.holist_models.gnn.gnn_encoder import GNNEncoder
from models.holist_models.transformer.transformer_encoder_model import TransformerWrapper as HOListTransformer
from models.ensemble.ensemble import EnsembleEmbedder

'''
Utility function to fetch model given a configuration dict
'''


def get_model(model_config):
    if model_config.model_type == 'sat':
        return GraphTransformer(in_size=model_config.model_attributes['vocab_size'],
                                num_class=2,
                                batch_norm=model_config.model_attributes[
                                    'batch_norm'] if 'batch_norm' in model_config.model_attributes else None,
                                d_model=model_config.model_attributes['embedding_dim'],
                                dim_feedforward=model_config.model_attributes['dim_feedforward'],
                                num_heads=model_config.model_attributes['num_heads'],
                                num_layers=model_config.model_attributes['num_layers'],
                                in_embed=model_config.model_attributes['in_embed'],
                                se=model_config.model_attributes['se'],
                                gnn_type=model_config.model_attributes[
                                    'gnn_type'] if 'gnn_type' in model_config.model_attributes else 'gcn',
                                abs_pe=model_config.model_attributes['abs_pe'],
                                abs_pe_dim=model_config.model_attributes['abs_pe_dim'],
                                use_edge_attr=model_config.model_attributes['use_edge_attr'],
                                num_edge_features=model_config.model_attributes['num_edge_features'],
                                global_pool=model_config.model_attributes['global_pool'],
                                dropout=model_config.model_attributes['dropout'],
                                k_hop=model_config.model_attributes['gnn_layers'],
                                small_inner=model_config.model_attributes[
                                    'small_inner'] if 'small_inner' in model_config.model_attributes else False)

    if model_config.model_type == 'amr':
        return AMRTransformer(in_size=model_config['vocab_size'],
                              d_model=model_config['embedding_dim'],
                              dim_feedforward=model_config['dim_feedforward'],
                              num_heads=model_config['num_heads'],
                              num_layers=model_config['num_layers'],
                              in_embed=model_config['in_embed'],
                              abs_pe=model_config['abs_pe'],
                              abs_pe_dim=model_config['abs_pe_dim'],
                              use_edge_attr=model_config['use_edge_attr'],
                              num_edge_features=200,  # model_config['num_edge_features'],
                              dropout=model_config['dropout'],
                              layer_norm=True,  # model_config['layer_norm'],
                              global_pool=True,  # model_config['global_pool'],
                              )


    elif model_config.model_type == 'formula-net':
        return FormulaNet(model_config['vocab_size'],
                          model_config['embedding_dim'],
                          model_config['gnn_layers'],
                          batch_norm=model_config['batch_norm'] if 'batch_norm' in model_config else True)

    elif model_config.model_type == 'formula-net-edges':
        return FormulaNetEdges(input_shape=model_config.model_attributes['vocab_size'],
                               embedding_dim=model_config.model_attributes['embedding_dim'],
                               num_iterations=model_config.model_attributes['gnn_layers'],
                               batch_norm=model_config.model_attributes[
                                   'batch_norm'] if 'batch_norm' in model_config.model_attributes else True)



    elif model_config.model_type == 'digae':
        return DigaeEmbedding(in_size=model_config.model_attributes['vocab_size'],
                              embedding_dim=model_config.model_attributes['embedding_dim'] // 2,
                              hidden_dim=model_config.model_attributes['embedding_dim'] // 2,
                              out_dim=model_config.model_attributes['embedding_dim'] // 2,
                              num_edges=model_config.model_attributes['num_edges'],
                              iterations=model_config.model_attributes['iterations']
                              )

    elif model_config.model_type == 'gcn':
        return GCNGNN(model_config.model_attributes['vocab_size'],
                      model_config.model_attributes['embedding_dim'],
                      model_config.model_attributes['gnn_layers'],
                      )

    elif model_config.model_type == 'di_gcn':
        return DiGCNGNN(model_config['vocab_size'],
                        model_config['embedding_dim'],
                        model_config['gnn_layers'],
                        )


    elif model_config.model_type == 'transformer':
        return TransformerWrapper(ntoken=model_config.model_attributes['vocab_size'],
                                  d_model=model_config.model_attributes['embedding_dim'],
                                  nhead=model_config.model_attributes['num_heads'],
                                  nlayers=model_config.model_attributes['num_layers'],
                                  dropout=model_config.model_attributes['dropout'],
                                  d_hid=model_config.model_attributes['dim_feedforward'],
                                  small_inner=model_config.model_attributes[
                                      'small_inner'] if 'small_inner' in model_config.model_attributes else False,
                                  max_len=model_config.model_attributes[
                                      'max_len'] if 'max_len' in model_config.model_attributes else 512)


    elif model_config.model_type == 'holist_transformer':
        return HOListTransformer(ntoken=model_config.model_attributes['vocab_size'],
                                 d_model=model_config.model_attributes['embedding_dim'],
                                 nhead=model_config.model_attributes['num_heads'],
                                 nlayers=model_config.model_attributes['num_layers'],
                                 dropout=model_config.model_attributes['dropout'],
                                 d_hid=model_config.model_attributes['dim_feedforward'],
                                 small_inner=model_config.model_attributes[
                                     'small_inner'] if 'small_inner' in model_config.model_attributes else False,
                                 max_len=model_config.model_attributes[
                                     'max_len'] if 'max_len' in model_config.model_attributes else 512)


    elif model_config.model_type == 'transformer_relation':
        return AttentionRelations(ntoken=model_config.model_attributes['vocab_size'],
                                  dropout=model_config.model_attributes[
                                      'dropout'] if 'dropout' in model_config.model_attributes else 0.0,
                                  num_heads=model_config.model_attributes[
                                      'num_heads'] if 'num_heads' in model_config.model_attributes else 8,
                                  num_layers=model_config.model_attributes[
                                      'num_layers'] if 'num_layers' in model_config.model_attributes else 4,
                                  embed_dim=model_config.model_attributes['embedding_dim'])

    elif model_config.model_type == 'transformer_relation_small':
        return AttentionRelationSmall(ntoken=model_config.model_attributes['vocab_size'],
                                      dropout=model_config.model_attributes[
                                          'dropout'] if 'dropout' in model_config.model_attributes else 0.0,
                                      num_heads=model_config.model_attributes[
                                          'num_heads'] if 'num_heads' in model_config.model_attributes else 8,
                                      num_layers=model_config.model_attributes[
                                          'num_layers'] if 'num_layers' in model_config.model_attributes else 4,
                                      embed_dim=model_config.model_attributes['embedding_dim'])

    elif model_config.model_type == 'gnn_transformer':
        return GNNTransformer(input_size=model_config['vocab_size'],
                              dropout=model_config['dropout'] if 'dropout' in model_config else 0.0,
                              num_heads=model_config['num_heads'] if 'num_heads' in model_config else 8,
                              num_layers=model_config['num_layers'] if 'num_layers' in model_config else 4,
                              embedding_dim=model_config['embedding_dim'],
                              batch_norm=model_config['batch_norm'],
                              dim_feedforward=model_config['dim_feedforward'],
                              num_iterations=model_config['gnn_layers'],
                              abs_pe=model_config['abs_pe'],
                              edge_dim=model_config['edge_dim'] if 'edge_dim' in model_config else 32,
                              max_edges=model_config['num_edges'] if 'num_edges' in model_config else 200,
                              global_pool=model_config['global_pool'] if 'global_pool' in model_config else 'mean',
                              )

    elif model_config.model_type == 'fixed_autoencoder':
        return TacticZeroAutoEncoder(model_config.model_attributes['checkpoint_path'])


    # holist models
    elif model_config.model_type == 'holist_gnn':
        return GNNEncoder(input_shape=model_config.model_attributes['vocab_size'],
                          embedding_dim=model_config.model_attributes['embedding_dim'],
                          num_iterations=model_config.model_attributes['gnn_layers'],
                          max_edges=model_config.model_attributes[
                              'max_edges'] if 'max_edges' in model_config.model_attributes else 3,
                          dropout=model_config.model_attributes[
                              'dropout'] if 'dropout' in model_config.model_attributes else 0.5)


    elif model_config.model_type == 'ensemble_holist':
        model_0 = GNNEncoder(input_shape=model_config.model_attributes['vocab_size'],
                             embedding_dim=model_config.model_attributes['embedding_dim'],
                             num_iterations=model_config.model_attributes['gnn_layers'],
                             dropout=model_config.model_attributes[
                                 'dropout'] if 'dropout' in model_config.model_attributes else 0.5)

        model_1 = HOListTransformer(ntoken=model_config.model_attributes['vocab_size'],
                                    d_model=model_config.model_attributes['embedding_dim'],
                                    nhead=model_config.model_attributes['num_heads'],
                                    nlayers=model_config.model_attributes['num_layers'],
                                    dropout=model_config.model_attributes['dropout'],
                                    d_hid=model_config.model_attributes['dim_feedforward'],
                                    small_inner=model_config.model_attributes[
                                        'small_inner'] if 'small_inner' in model_config.model_attributes else False,
                                    max_len=model_config.model_attributes[
                                        'max_len'] if 'max_len' in model_config.model_attributes else 512)

        return EnsembleEmbedder(d_model=model_config.model_attributes['embedding_dim'] * 8,
                                gnn_model=model_0,
                                transformer_model=model_1,
                                dropout=model_config.model_attributes['dropout'])


    elif model_config.model_type == 'ensemble':
        model_0 = FormulaNetEdges(input_shape=model_config.model_attributes['vocab_size'],
                                  embedding_dim=model_config.model_attributes['embedding_dim'],
                                  num_iterations=model_config.model_attributes['gnn_layers'],
                                  batch_norm=model_config.model_attributes[
                                      'batch_norm'] if 'batch_norm' in model_config.model_attributes else True)

        model_1 = TransformerWrapper(ntoken=model_config.model_attributes['vocab_size'],
                                     d_model=model_config.model_attributes['embedding_dim'],
                                     nhead=model_config.model_attributes['num_heads'],
                                     nlayers=model_config.model_attributes['num_layers'],
                                     dropout=model_config.model_attributes['dropout'],
                                     d_hid=model_config.model_attributes['dim_feedforward'],
                                     small_inner=model_config.model_attributes[
                                         'small_inner'] if 'small_inner' in model_config.model_attributes else False,
                                     max_len=model_config.model_attributes[
                                         'max_len'] if 'max_len' in model_config.model_attributes else 512)

        return EnsembleEmbedder(d_model=model_config.model_attributes['embedding_dim'],
                                gnn_model=model_0,
                                transformer_model=model_1,
                                dropout=model_config.model_attributes['dropout'])


    elif model_config.model_type == 'holist_sat':
        return HOListSAT(in_size=model_config.model_attributes['vocab_size'],
                         num_class=2,
                         batch_norm=model_config.model_attributes[
                             'batch_norm'] if 'batch_norm' in model_config.model_attributes else None,
                         d_model=model_config.model_attributes['embedding_dim'],
                         dim_feedforward=model_config.model_attributes['dim_feedforward'],
                         num_heads=model_config.model_attributes['num_heads'],
                         num_layers=model_config.model_attributes['num_layers'],
                         in_embed=model_config.model_attributes['in_embed'],
                         se=model_config.model_attributes['se'],
                         gnn_type=model_config.model_attributes[
                             'gnn_type'] if 'gnn_type' in model_config.model_attributes else 'gcn',
                         abs_pe=model_config.model_attributes['abs_pe'],
                         abs_pe_dim=model_config.model_attributes['abs_pe_dim'],
                         use_edge_attr=model_config.model_attributes['use_edge_attr'],
                         num_edge_features=model_config.model_attributes['num_edge_features'],
                         global_pool=model_config.model_attributes['global_pool'],
                         dropout=model_config.model_attributes['dropout'],
                         k_hop=model_config.model_attributes['gnn_layers'],
                         small_inner=model_config.model_attributes[
                             'small_inner'] if 'small_inner' in model_config.model_attributes else False)

    elif model_config.model_type == 'classifier':
        raise NotImplementedError
    else:
        return None
