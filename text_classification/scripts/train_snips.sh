#!/bin/bash

for test_seed in 0 1 2 3 4 5 6 7 8 9
do
    for test_id_ratio in 0.25 0.50 0.75
    do
        test_n_labels=$(printf "%.0f"  $(echo  "16 * $test_id_ratio" | bc))
        echo "# of test labels: $test_n_labels"

        python -m text_classification.train_text_classifier \
            --config text_classification/configs/snips_test.yml \
            --trainer.logger.init_args.name snips_classifier_ts_${test_seed}_tr_${test_id_ratio} \
            --model.num_labels $test_n_labels \
            --model.checkpoint_path checkpoints/snips_cs/ts_${test_seed}_tr_${test_id_ratio} \
            --data.init_args.class_split_seed $test_seed \
            --data.init_args.seen_class_ratio $test_id_ratio 

        for val_seed in 0
        do
            for val_id_ratio in 0.25 0.5 0.75
            do
                val_n_labels=$(printf "%.0f"  $(echo  "$test_n_labels * $val_id_ratio" | bc))
                echo "# of val labels: $val_n_labels"

                if [ $val_n_labels -lt 2 ]; then
                    echo "Skipping # of ID val labels: $val_n_labels"
                    continue
                fi

                python -m text_classification.train_text_classifier \
                    --config text_classification/configs/snips_val.yml \
                    --trainer.logger.init_args.name snips_classifier_ts_${test_seed}_tr_${test_id_ratio}_vs_${val_seed}_vr_${val_id_ratio} \
                    --model.num_labels $val_n_labels \
                    --model.checkpoint_path checkpoints/snips_cs/ts_${test_seed}_tr_${test_id_ratio}_vs_${val_seed}_vr_${val_id_ratio} \
                    --data.init_args.class_split_seed $val_seed \
                    --data.init_args.seen_class_ratio $val_id_ratio \
                    --data.init_args.train_dataset.init_args.class_split_seed $test_seed \
                    --data.init_args.val_dataset.init_args.class_split_seed $test_seed \
                    --data.init_args.train_dataset.init_args.seen_class_ratio $test_id_ratio \
                    --data.init_args.val_dataset.init_args.seen_class_ratio $test_id_ratio
            done
        done

    done
done