# Split SHREC_2017 dataset training set into training and validation sets

# dataset dir
SHREC2017_root_dir=/.../SHREC2017
# code dir
project_dir=/.../IR-STDGCN

cd ${project_dir}/utils/preprocess
python SHREC2017_split_train_to_train_val.py --root_dir ${SHREC2017_root_dir}
