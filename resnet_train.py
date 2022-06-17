from models.model import Model
from models.fashion_mnist_fc import FashionMnistModel
from models.resnet_16 import Resnet16Model
from utils.model_state import ModelState
from utils.model_training import ModelTraining
from utils.data_distribution import PartitioningScheme
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1

import os
import json
import random
import numpy as np
import tensorflow as tf
import pickle

import utils.model_merge as merge_ops
import utils.model_purge as purge_ops

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
np.random.seed(1990)
random.seed(1990)
tf.random.set_seed(1990)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":

    """ Model Definition. """
    lambda1 = l1(0.0001)
    lambda2 = l2(0.0001)

    model = Resnet16Model(kernel_initializer=Model.InitializationStates.HE_NORMAL, learning_rate=0.02,
                              use_sgd=True, use_fedprox=False, use_sgd_with_momentum=False, fedprox_mu=0.0,
                              momentum_factor=0.0, kernel_regularizer=None, bias_regularizer=None).get_model
    model().summary()

    """ Load the data. """
    with open("./data/RML2016.10b.dat", "rb") as p:
        Xd = pickle.load(p, encoding='latin1')
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    print("length of snr",len(snrs))
    print("length of mods",len(mods))
    X = [] 
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
    X = np.vstack(X)
    print("shape of X", np.shape(X))

    """ Partition the dataset into training and testing datasets """
    np.random.seed(2016)     # Random seed value for the partitioning (Also used for random subsampling)
    n_examples = X.shape[0]
    n_train = n_examples // 2
    train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0,n_examples))-set(train_idx))
    x_train = X[train_idx]
    x_test =  X[test_idx]
    def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy)+1])
        yy1[np.arange(len(yy)),yy] = 1
        return yy1
    y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))


    output_logs_dir = "./logs/"
    output_npzarrays_dir = "./npzarrays/"
    experiment_template = \
        "Resnet16.rounds_{}.learners_{}.participation_{}.le_{}.compression_{}.sparsificationround_{}.sparsifyevery_{}rounds.finetuning_{}"

    rounds_num = 100
    learners_num_list = [1]
    participation_rates_list = [1]

    # One-Shot Pruning
    # sparsity_levels = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
    # sparsity_levels = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
    # start_sparsification_at_round = [1, 5, 10, 90]

    # Federated Progressive Pruning
    sparsity_levels = [0.99]
    sparsification_frequency = [1]
    start_sparsification_at_round = [1]

    local_epochs = 1
    fine_tuning_epochs = [0]
    batch_size = 32
    train_with_global_mask = True

    for learners_num, participation_rate  in zip(learners_num_list, participation_rates_list):
        for sparsity_level in sparsity_levels:
            for frequency in sparsification_frequency:
                for sparsification_round in start_sparsification_at_round:
                    for fine_tuning_epoch_num in fine_tuning_epochs:

                        # fill in string placeholders
                        filled_in_template = experiment_template.format(rounds_num,
                                                                        learners_num,
                                                                        str(participation_rate).replace(".", ""),
                                                                        str(local_epochs),
                                                                        str(sparsity_level).replace(".", ""),
                                                                        str(sparsification_round),
                                                                        str(frequency),
                                                                        fine_tuning_epoch_num)
                        output_arrays_dir = output_npzarrays_dir + filled_in_template

                        pscheme = PartitioningScheme(x_train=x_train, y_train=y_train, partitions_num=learners_num)
                        x_chunks, y_chunks = pscheme.iid_partition()
                        #x_chunks, y_chunks = pscheme.non_iid_partition(classes_per_partition=2)

                        scaling_factors = [y_chunk.size for y_chunk in y_chunks]

                        # Merging Ops.
                        merge_op = merge_ops.MergeWeightedAverage(scaling_factors)
                        # merge_op = merge_ops.MergeMedian(scaling_factors)
                        # merge_op = merge_ops.MergeAbsMax(scaling_factors)
                        # merge_op = merge_ops.MergeAbsMin(scaling_factors, discard_zeroes=True)
                        # merge_op = merge_ops.MergeTanh(scaling_factors)
                        # merge_op = merge_ops.MergeWeightedAverageNNZ(scaling_factors)
                        # merge_op = merge_ops.MergeWeightedAverageMajorityVoting(scaling_factors)
                        # merge_op = merge_ops.MergeWeightedPseudoGradients(scaling_factors)

                        # Purging Ops.
                        # purge_op = purge_ops.PurgeByWeightMagnitude(sparsity_level=sparsity_level)
                        # purge_op = purge_ops.PurgeByNNZWeightMagnitude(sparsity_level=sparsity_level,
                        # 											   sparsify_every_k_round=frequency)
                        # purge_op = purge_ops.PurgeByNNZWeightMagnitudeRandom(sparsity_level=sparsity_level,
                        # 													 num_params=model().count_params(),
                        # 													 sparsify_every_k_round=frequency)
                        # purge_op = purge_ops.PurgeByLayerWeightMagnitude(sparsity_level=sparsity_level)
                        # purge_op = purge_ops.PurgeByLayerNNZWeightMagnitude(sparsity_level=sparsity_level)
                        # purge_op = purge_ops.PurgeByWeightMagnitudeRandomGradual(num_params=model().count_params(),
                        # 														 start_at_round=sparsification_round,
                        # 														 sparsity_level_init=0.0,
                        # 														 sparsity_level_final=sparsity_level,
                        # 														 total_rounds=rounds_num,
                        # 														 delta_round_pruning=frequency,
                        # 														 exponent=3,
                        # 														 federated_model=True)
                        purge_op = purge_ops.PurgeByWeightMagnitudeGradual(start_at_round=sparsification_round,
                                                                           sparsity_level_init=0.0,
                                                                            sparsity_level_final=sparsity_level,
                                                                            total_rounds=rounds_num,
                                                                            delta_round_pruning=frequency,
                                                                            exponent=3)
                        # 												   federated_model=True)

                        # sparsity_level = purge_op.to_json()
                        # randint = random.randint(0, learners_num-1)
                        # purge_op = purge_ops.PurgeSNIP(model(),
                        # 							   sparsity=sparsity_level,
                        # 							   x=x_chunks[randint][:batch_size],
                        # 							   y=y_chunks[randint][:batch_size])
                        #randint = random.randint(0, learners_num-1)
                        #purge_op = purge_ops.PurgeGrasp(model(),
                        #							   sparsity=sparsity_level,
                        #							   x=x_chunks[randint][:batch_size],
                        #							   y=y_chunks[randint][:batch_size])

                        federated_training = ModelTraining.FederatedTraining(merge_op=merge_op,
                                                                             learners_num=learners_num,
                                                                             rounds_num=rounds_num,
                                                                             local_epochs=local_epochs,
                                                                             learners_scaling_factors=scaling_factors,
                                                                             participation_rate=participation_rate,
                                                                             batch_size=batch_size,
                                                                             purge_op_local=None,
                                                                             purge_op_global=purge_op,
                                                                             start_purging_at_round=sparsification_round,
                                                                             fine_tuning_epochs=fine_tuning_epoch_num,
                                                                             train_with_global_mask=train_with_global_mask,
                                                                             start_training_with_global_mask_at_round=sparsification_round,
                                                                             output_arrays_dir=output_arrays_dir)
                        #													 precomputed_masks=purge_op.precomputed_masks)
                        federated_training.execution_stats['federated_environment']['model_params'] = ModelState.count_non_zero_elems(model())
                        federated_training.execution_stats['federated_environment']['sparsity_level'] = sparsity_level
                        federated_training.execution_stats['federated_environment']['additional_specs'] = purge_op.json()
                        federated_training.execution_stats['federated_environment']['data_distribution'] = \
                            pscheme.to_json_representation()
                        federated_training_results = federated_training.start(get_model_fn=model, x_train_chunks=x_chunks,
                                                                              y_train_chunks=y_chunks, x_test=x_test,
                                                                              y_test=y_test, info="Resnet16")

                        execution_output_filename = output_logs_dir + filled_in_template + ".json"
                        with open(execution_output_filename, "w+", encoding='utf-8') as fout:
                            json.dump(federated_training_results, fout, ensure_ascii=False, indent=4)