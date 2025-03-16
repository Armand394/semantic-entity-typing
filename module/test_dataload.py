import argparse
from utils import *
from dataloader import SEMdataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

device = torch.device('cuda:0')

def main(args):
    use_cuda = args['cuda'] and torch.cuda.is_available()
    data_path = os.path.join(args['data_dir'], args['dataset'])

    e2id = read_id(os.path.join(data_path, 'entities.tsv'))
    r2id = read_id(os.path.join(data_path, 'relations.tsv'))
    t2id = read_id(os.path.join(data_path, 'types.tsv'))
    c2id = read_id(os.path.join(data_path, 'clusters.tsv'))
    e2desc, e2text = read_entity_wiki(os.path.join(data_path, 'entity_wiki.json'), e2id, args['semantic'])
    r2text = read_rel_context(os.path.join(data_path, 'relation2text.txt'), r2id)
    t2desc = read_type_context(os.path.join(data_path, 'hier_type_desc.txt'), t2id)
    num_entities = len(e2id)
    print(num_entities)
    num_rels = len(r2id)
    num_types = len(t2id)
    num_clusters = len(c2id)
    train_type_label, test_type_label = load_train_all_labels(data_path, e2id, t2id)
    if use_cuda:
        sample_ent2pair = torch.LongTensor(load_entity_cluster_type_pair_context(args, r2id, e2id)).cuda()
    train_dataset = SEMdataset(args, "KG_train.txt", e2id, r2id, t2id, c2id, 'train')


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='FB15kET')
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('--save_path', type=str, default='SFNA')
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--valid_epoch', type=int, default=25)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--plm', type=str, default='bert-base-uncased')
    parser.add_argument('--loss', type=str, default='SFNA')

    # params for first trm layer
    parser.add_argument('--bert_nlayer', type=int, default=3)
    parser.add_argument('--bert_nhead', type=int, default=4)
    parser.add_argument('--bert_ff_dim', type=int, default=480)
    parser.add_argument('--bert_activation', type=str, default='gelu')
    parser.add_argument('--bert_hidden_dropout', type=float, default=0.2)
    parser.add_argument('--bert_attn_dropout', type=float, default=0.2)
    parser.add_argument('--local_pos_size', type=int, default=200)

    # params for pair trm layer
    parser.add_argument('--pair_layer', type=int, default=3)
    parser.add_argument('--pair_head', type=int, default=4)
    parser.add_argument('--pair_dropout', type=float, default=0.2)
    parser.add_argument('--pair_ff_dim', type=int, default=480)

    # params for second trm layer
    parser.add_argument('--trm_nlayer', type=int, default=3)
    parser.add_argument('--trm_nhead', type=int, default=4)
    parser.add_argument('--trm_hidden_dropout', type=float, default=0.2)
    parser.add_argument('--trm_attn_dropout', type=float, default=0.2)
    parser.add_argument('--trm_ff_dim', type=int, default=480)
    parser.add_argument('--global_pos_size', type=int, default=200)

    parser.add_argument('--pair_pooling', type=str, default='avg', choices=['max', 'avg', 'min'])
    parser.add_argument('--sample_et_size', type=int, default=3)
    parser.add_argument('--sample_kg_size', type=int, default=7)
    parser.add_argument('--sample_ent2pair_size', type=int, default=6)
    parser.add_argument('--warm_up_steps', default=50, type=int)
    parser.add_argument('--tt_ablation', type=str, default='all', choices=['all', 'triple', 'type'],
                        help='ablation choice')
    parser.add_argument('--log_name', type=str, default='log')
    parser.add_argument('--semantic', type=str, default='hybrid')

    args, _ = parser.parse_known_args()
    print(args)
    return args


if __name__ == '__main__':
    try:
        params = vars(get_params())
        set_logger(params)
        main(params)
    except Exception as e:
        logging.exception(e)
        raise
