# dataset dir
SHREC2017_root_dir=/.../SHREC2017
# code dir
project_dir=/.../IR-STDGCN

cd ${project_dir}/utils/preprocess
python gen_SHREC2017_dataset.py --root_dir ${SHREC2017_root_dir}
