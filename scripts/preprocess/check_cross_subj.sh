# Check cross-subject dataset split of SHREC_2017 dataset

# dataset dir
SHREC2017_root_dir=/.../IR-STDGCN/SHREC2017
# code dir
project_dir=/.../IR-STDGCN

cd ${project_dir}/utils/preprocess
python check_cross_subj_SHREC2017.py --root_dir ${SHREC2017_root_dir}
