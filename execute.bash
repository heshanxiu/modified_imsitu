#The name of the experiment
name=$2

output=snap/$name
mkdir -p $output/src
cp -r *.py $output/src
cp $0 $output/execute.bash

CUDA_VISIBLE_DEVICES=$1 python baseline_crf.py --command train_on_dev\
 --output_dir $output \
 --dataset_dir ./visualsrl_data_dev\
 --encoding_file baseline_models/baseline_encoder \
 --batch_size 1 \
 --image_dir resized_256 \
 --eval_file ./visualsrl_data\
 --extract_feature True \
 --feature_level 5


#CUDA_VISIBLE_DEVICES=$1 python eval.py --format model \
# --include baseline_crf.py \
# --weights_file $output/*model\
# --encoding_file baseline_models/baseline_encoder \
# --trust_encoder \
# --batch_size 1 \
# --image_dir resized_256 \
# --extract_feature True \
# --feature_level 5