# dataset dir
SHREC2017_root_dir=/.../SHREC2017
# code dir
project_dir=/.../IR-STDGCN

cd ${project_dir}/utils/preprocess
python get_class_freq.py \
--root_dir ${SHREC2017_root_dir} \
--dataset SHREC2017
