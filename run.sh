echo "job start time: `date`"
CUDA_VISIBLE_DEVICES=1 /Work21/2021/liyuhang/envs/py3.7/dar_env/bin/python engine.py --config_file ./config/swda/swda.yaml
echo "job end time: `date`"